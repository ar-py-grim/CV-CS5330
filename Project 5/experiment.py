# Arpit Gandhi
# April 2026
# Automated coordinate-descent hyperparameter search for NetTransformer on MNIST.

import os, time, csv, itertools
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from NetTransformer import NetTransformer, NetConfig

# Paths & constants
BASE_DIR = os.path.dirname(__file__)
RESULTS_FILE = os.path.join(BASE_DIR, "models.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
DATA_DIR = os.path.join(BASE_DIR, "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(PLOTS_DIR, exist_ok=True)

# Default hyperparameters phases override only what they vary
BASELINE = dict(
    dataset="mnist", patch_size=7, stride=7, embed_dim=48, depth=4,
    num_heads=4, mlp_dim=128, dropout=0.1, use_cls_token=False,
    epochs=10, batch_size=64, lr=1e-3, weight_decay=1e-4,
    seed=42, optimizer="adamw", device=DEVICE,
)

CSV_FIELDS = [
    "run_id", "phase", "name", "dataset",
    "patch_size", "stride", "embed_dim", "depth", "num_heads",
    "mlp_dim", "dropout", "use_cls_token",
    "epochs", "batch_size", "lr", "weight_decay",
    "test_acc", "best_epoch", "avg_epoch_s", "n_params",
]

# Data
def get_loaders(batch_size: int):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(DATA_DIR, train=True,  download=False, transform=tfm)
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=False, transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=(DEVICE!= "cpu"))
    test_dl = DataLoader(test_ds,  batch_size=256, shuffle=False,
                          num_workers=2, pin_memory=(DEVICE!= "cpu"))
    
    return train_dl, test_dl

# Training
def run_experiment(cfg: NetConfig) -> dict:
    torch.manual_seed(cfg.seed)
    train_dl, test_dl = get_loaders(cfg.batch_size)

    model = NetTransformer(cfg).to(cfg.device)
    opt = (optim.AdamW if cfg.optimizer == "adamw" else optim.Adam)(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_acc, best_epoch, epoch_times = 0.0, 0, []

    for epoch in range(1, cfg.epochs+1):
        t0 = time.time()
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            opt.zero_grad()
            nn.functional.nll_loss(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                correct+= (model(xb).argmax(1) == yb).sum().item()
                total+= yb.size(0)

        acc = correct/total
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
        epoch_times.append(time.time()-t0)
        print(f"    epoch {epoch:2d}/{cfg.epochs}  acc={acc:.4f}  "
              f"best={best_acc:.4f}  t={epoch_times[-1]:.1f}s")

    return {
        "test_acc": round(best_acc, 5),
        "best_epoch": best_epoch,
        "avg_epoch_s": round(sum(epoch_times)/len(epoch_times), 2),
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

# Logging into CSV
def init_csv():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

def log_result(run_id: int, phase: str, cfg: NetConfig, metrics: dict):
    row = {
        "run_id": run_id, "phase": phase, "name": cfg.name, "dataset": cfg.dataset,
        "patch_size": cfg.patch_size, "stride": cfg.stride, "embed_dim": cfg.embed_dim,
        "depth": cfg.depth, "num_heads": cfg.num_heads, "mlp_dim": cfg.mlp_dim,
        "dropout": cfg.dropout, "use_cls_token": cfg.use_cls_token,
        "epochs": cfg.epochs, "batch_size": cfg.batch_size,
        "lr": cfg.lr, "weight_decay": cfg.weight_decay, **metrics,
    }
    with open(RESULTS_FILE, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
    print(f" logged run {run_id}  acc={metrics['test_acc']:.4f}\n")

# Config helpers
def make_cfg(name: str, overrides: dict) -> NetConfig:
    return NetConfig(**{**BASELINE, **overrides, "name": name})

def best_cfg(models: list) -> NetConfig:
    return max(models, key=lambda r: r["test_acc"])["cfg"]

# Phase builders
def phase1_depth():
    return [make_cfg(f"d{d}", {"depth": d}) for d in [1, 2, 4, 6, 8]]

def phase2_capacity(depth):
    specs = [(32,64,4), (48,128,4), (64,192,4), (96,256,4), (128,384,8)]
    return [make_cfg(f"e{e}_m{m}", {"depth": depth, "embed_dim": e, "mlp_dim": m, "num_heads": h})
            for e, m, h in specs]

def phase3_patch(depth, embed_dim, mlp_dim, num_heads):
    base = {"depth": depth, "embed_dim": embed_dim, "mlp_dim": mlp_dim, "num_heads": num_heads}
    pairs = [(14,7), (7,7), (7,4), (4,4), (7,2), (4,2)]
    return [make_cfg(f"p{ps}s{st}", {**base, "patch_size": ps, "stride": st}) for ps, st in pairs]

def phase4_depth(patch_size, stride, embed_dim, mlp_dim, num_heads):
    base = {"patch_size": patch_size, "stride": stride,
            "embed_dim": embed_dim, "mlp_dim": mlp_dim, "num_heads": num_heads}
    return [make_cfg(f"d{d}_p{patch_size}s{stride}", {**base, "depth": d}) for d in [1, 2, 4, 6, 8]]

def phase5_regularization(best_ov: dict):
    runs = [make_cfg(f"dr{dr}_mean", {**best_ov, "dropout": dr, "use_cls_token": False})
             for dr in [0.0, 0.05, 0.1, 0.2, 0.3]]
    runs+= [make_cfg(f"dr{dr}_cls",  {**best_ov, "dropout": dr, "use_cls_token": True})
             for dr in [0.1, 0.2]]
    return runs

def phase6_grid(best_ov: dict):
    d_opts = sorted({max(1, best_ov["depth"] - 2), best_ov["depth"], best_ov["depth"] + 2})
    e_opts = [e for e in [best_ov["embed_dim"] - 16, best_ov["embed_dim"], best_ov["embed_dim"] + 16] if e >= 16]
    return [
        make_cfg(f"grid_d{d}_e{e}", {
            **best_ov, "depth": d, "embed_dim": e, "mlp_dim": e * 3,
            "num_heads": 8 if e >= 96 else 4,
        })
        for d, e in itertools.product(d_opts, e_opts)
        if not (d == best_ov["depth"] and e == best_ov["embed_dim"])
    ]

def phase7_heads(best_ov: dict):
    return [
        make_cfg(f"h{h}", {**best_ov, "num_heads": h})
        for h in [1, 2, 4, 8]
        if h!= best_ov["num_heads"] and best_ov["embed_dim"] % h == 0
    ]

# Plotting

def save_fig(fig, fname: str):
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")

def annotate_points(ax, x, y):
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.2f}%", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)

def plot_depth_phase1(df):
    sub = df[df["phase"] == "p1_depth"].sort_values("depth")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sub["depth"], sub["test_acc_pct"], "o-", color="steelblue", lw=2, ms=8)
    annotate_points(ax, sub["depth"], sub["test_acc_pct"])
    ax.set(xlabel="Depth", ylabel="Test Accuracy (%)",
           title="Depth Sweep\n(embed=48, patch=7/7, dropout=0.1)")
    ax.set_xticks(sub["depth"]); ax.grid(True, alpha=0.3)
    save_fig(fig, "1a_depth_phase1.png")

def plot_depth_comparison(df):
    p1 = df[df["phase"] == "p1_depth"].sort_values("depth")
    p4 = df[df["phase"] == "p4_depth_refined"].sort_values("depth")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(p1["depth"], p1["test_acc_pct"], "o--", color="steelblue", lw=2, ms=7, label="Phase 1 (base patch)")
    ax.plot(p4["depth"], p4["test_acc_pct"], "s-",  color="navy",      lw=2, ms=7, label="Phase 4 (best patch)")
    ax.set(xlabel="Depth", ylabel="Test Accuracy (%)",
           title="Depth: Before vs After Patch Optimization")
    ax.set_xticks(sorted(set(p1["depth"].tolist() + p4["depth"].tolist())))
    ax.legend(); ax.grid(True, alpha=0.3)
    save_fig(fig, "1b_depth_comparison.png")

def plot_capacity(df):
    sub = df[df["phase"] == "p2_capacity"].sort_values("embed_dim")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sub["embed_dim"], sub["test_acc_pct"], "s-", color="tomato", lw=2, ms=8)
    annotate_points(ax, sub["embed_dim"], sub["test_acc_pct"])
    ax2 = ax.twinx()
    ax2.plot(sub["embed_dim"], sub["n_params"] / 1000, "^:", color="grey", lw=1.5, ms=6, alpha=0.6)
    ax2.set_ylabel("# Parameters (K)", color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")
    ax.set(xlabel="Embedding Dimension", ylabel="Test Accuracy (%)",
           title="Capacity Sweep\n(depth=best, patch=7/7, dropout=0.1)")
    ax.set_xticks(sub["embed_dim"]); ax.grid(True, alpha=0.3)
    save_fig(fig, "2_capacity_sweep.png")

def plot_patch_geometry(df):
    sub = df[df["phase"] == "p3_patch"].copy()
    sub["label"]  = sub["patch_size"].astype(str) + "/" + sub["stride"].astype(str)
    sub["tokens"] = ((28-sub["patch_size"])//sub["stride"]+1)**2
    sub = sub.sort_values("tokens")
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(sub["label"], sub["test_acc_pct"], color="mediumseagreen", edgecolor="white", width=0.6)
    for bar, (_, row) in zip(bars, sub.iterrows()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                f"{row['test_acc_pct']:.2f}%\n({int(row['tokens'])} tok)",
                ha="center", va="bottom", fontsize=8)
    ax.set(xlabel="Patch Size / Stride  (sorted by token count)",
           ylabel="Test Accuracy (%)",
           title="Patch Geometry Sweep")
    ax.set_ylim(sub["test_acc_pct"].min()-1, sub["test_acc_pct"].max()+2)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "3_patch_geometry.png")

def plot_regularization(df):
    sub = df[df["phase"] == "p5_regularization"].copy()
    fig, ax = plt.subplots(figsize=(6, 4))
    for cls_val, grp in sub.groupby("use_cls_token"):
        grp = grp.sort_values("dropout")
        label  = "CLS token" if cls_val else "Mean pooling"
        marker = "D-" if cls_val else "o-"
        ax.plot(grp["dropout"], grp["test_acc_pct"], marker, lw=2, ms=8, label=label)
        annotate_points(ax, grp["dropout"], grp["test_acc_pct"])
    ax.set(xlabel="Dropout Rate", ylabel="Test Accuracy (%)",
           title="Dropout x Pooling Strategy")
    ax.legend(); ax.grid(True, alpha=0.3)
    save_fig(fig, "4_regularization.png")

def plot_heads(df):
    sub = df[df["phase"] == "p7_heads"].sort_values("num_heads")
    if sub.empty:
        print("No Phase 7 data to plot")
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(sub["num_heads"], sub["test_acc_pct"], "o-", color="mediumpurple", lw=2, ms=8)
    annotate_points(ax, sub["num_heads"], sub["test_acc_pct"])
    ax.set(xlabel="Number of Attention Heads", ylabel="Test Accuracy (%)",
           title="Attention Head Sweep")
    ax.set_xticks(sub["num_heads"]); ax.grid(True, alpha=0.3)
    save_fig(fig, "5_num_heads.png")

# Analysis

def run_analysis():
    df = pd.read_csv(RESULTS_FILE)
    df["test_acc_pct"] = df["test_acc"]*100

    # Best per phase
    print("\nBEST PER PHASE")
    for phase, grp in df.groupby("phase"):
        b = grp.loc[grp["test_acc"].idxmax()]
        print(f"  {phase:<24} acc={b['test_acc_pct']:.3f}%  depth={int(b['depth'])}  "
              f"embed={int(b['embed_dim'])}  patch={int(b['patch_size'])}/{int(b['stride'])}  "
              f"drop={b['dropout']}")

    # Top 10 overall
    cols = ["run_id","phase","depth","embed_dim","patch_size","stride",
            "dropout","use_cls_token","num_heads","test_acc_pct","n_params"]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 130)
    print(f"\nTOP 10 OVERALL\n")
    print(df.nlargest(10, "test_acc")[cols].to_string(index=False))
    print(f"\nTotal runs: {len(df)}  |  "
          f"Range: {df['test_acc_pct'].min():.2f}% - {df['test_acc_pct'].max():.2f}%")

    plot_depth_phase1(df)
    plot_depth_comparison(df)
    plot_capacity(df)
    plot_patch_geometry(df)
    plot_regularization(df)
    plot_heads(df)

# main function
def main():
    init_csv()
    all_results = []
    run_id = 0

    def run_phase(configs: list, label: str, phase_tag: str) -> list:
        nonlocal run_id
        print(f"\n{label}\n")
        phase_results = []
        for cfg in configs:
            run_id+= 1
            print(f"\n[Run {run_id}] {cfg.name}")
            metrics = run_experiment(cfg)
            log_result(run_id, phase_tag, cfg, metrics)
            record = {"cfg": cfg, **metrics}
            all_results.append(record)
            phase_results.append(record)
        return phase_results

    # Coordinate descent
    p1 = run_phase(phase1_depth(), "Phase 1 — Depth", "p1_depth")
    d = best_cfg(p1).depth

    p2 = run_phase(phase2_capacity(d), "Phase 2 — Capacity", "p2_capacity")
    b2 = best_cfg(p2)
    e, m, h = b2.embed_dim, b2.mlp_dim, b2.num_heads

    p3 = run_phase(phase3_patch(d, e, m, h), "Phase 3 — Patch Geometry", "p3_patch")
    b3 = best_cfg(p3)
    p, s = b3.patch_size, b3.stride

    p4 = run_phase(phase4_depth(p, s, e, m, h), "Phase 4 — Depth (Refined)", "p4_depth_refined")
    d = best_cfg(p4).depth

    best_ov = {"depth": d, "patch_size": p, "stride": s, "embed_dim": e, "mlp_dim": m, "num_heads": h}
    p5 = run_phase(phase5_regularization(best_ov), "Phase 5 — Regularization", "p5_regularization")
    b5 = best_cfg(p5)
    best_ov.update({"dropout": b5.dropout, "use_cls_token": b5.use_cls_token})

    p6 = run_phase(phase6_grid(best_ov), "Phase 6 — Grid Around Best", "p6_grid")
    b6 = best_cfg(p6 + [max(p5, key=lambda r: r["test_acc"])])
    best_ov.update({"depth": b6.depth, "embed_dim": b6.embed_dim, "mlp_dim": b6.mlp_dim})

    run_phase(phase7_heads(best_ov), "Phase 7 — Attention Heads", "p7_heads")

    ob = max(all_results, key=lambda r: r["test_acc"])
    bc = ob["cfg"]
    print(f"FINAL BEST  acc={ob['test_acc']:.4f}  depth={bc.depth}  embed={bc.embed_dim}  "
          f"patch={bc.patch_size}/{bc.stride}  drop={bc.dropout}  "
          f"cls={bc.use_cls_token}  params={ob['n_params']:,}")

    run_analysis()


if __name__ == "__main__":
    main()
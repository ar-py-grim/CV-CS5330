# Arpit Gandhi
# February 2026

# GUI application using Tkinter to manage real time 2D object recognition system.
# It allows the user to launch/stop the C++ recognition app, view and manage database

import tkinter as tk
from tkinter import ttk, messagebox
import os
import subprocess
import threading

# paths
PROJECT_EXE = os.path.join("x64", "Debug", "Project3.exe")
HC_DB  = "object_db.csv"
CNN_DB = "cnn_db.csv"
CM_HC  = "confusion_hc.txt"
CM_CNN = "confusion_cnn.txt"

# dimensions
HC_DIM  = 9
CNN_DIM = 512

# colors
BG      = "#1e1e2e"
PANEL   = "#2a2a3e"
ACCENT  = "#7c83fd"
TEXT    = "#cdd6f4"
SUBTEXT = "#6c7086"
GREEN   = "#a6e3a1"
RED     = "#f38ba8"
YELLOW  = "#f9e2af"
BORDER  = "#45475a"

# load CSV files
def load_csv(path, expected_dim):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) == expected_dim + 1:
                rows.append(parts)
    return rows

# saving CSV files
def save_csv(path, rows):
    with open(path, "w", newline="") as f:
        for r in rows:
            f.write(",".join(r) + "\n")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Object Recognition Manager")
        self.configure(bg=BG)
        self.geometry("1300x750")
        self.resizable(True, True)
        self._proc = None                      
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()

    # Clean shutdown
    def _on_close(self):
        self._kill_proc()
        self.destroy()

    def _kill_proc(self):
        if self._proc and self._proc.poll() is None: 
            self._proc.kill()
            self._proc = None

    # building UI
    def _build_ui(self):
        self.status_var = tk.StringVar(value="Ready")

        tk.Label(self, text="Object Recognition System",
                 bg=BG, fg=ACCENT,
                 font=("Segoe UI", 16, "bold"), pady=8).pack(fill="x")

        content = tk.Frame(self, bg=BG)
        content.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self._build_left_panel(content)
        self._build_tabs(content)

        tk.Label(self, textvariable=self.status_var,
                 bg=PANEL, fg=SUBTEXT,
                 font=("Segoe UI", 9), anchor="w", padx=10, pady=4).pack(fill="x", side="bottom")

    def _build_left_panel(self, parent):
        frame = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER,
                         highlightthickness=1, width=260)
        frame.pack(side="left", fill="y", padx=(0, 8))
        frame.pack_propagate(False)

        tk.Label(frame, text="Controls", bg=PANEL, fg=ACCENT,
                 font=("Segoe UI", 11, "bold"), pady=12).pack()

        tk.Label(frame, text="UI for live object recognition.",
                 bg=PANEL, fg=SUBTEXT, font=("Segoe UI", 9),
                 justify="center").pack(pady=(0, 16))

        self.launch_btn = tk.Button(frame, text="▶  Launch Recognition",
                                    command=self._launch_cpp,
                                    bg=GREEN, fg=BG,
                                    font=("Segoe UI", 10, "bold"),
                                    relief="flat", padx=14, pady=8,
                                    cursor="hand2", width=20)
        self.launch_btn.pack(pady=(0, 8))

        self.stop_btn = tk.Button(frame, text="■  Stop Recognition",
                                  command=self._stop_cpp,
                                  bg=RED, fg=BG,
                                  font=("Segoe UI", 10, "bold"),
                                  relief="flat", padx=14, pady=8,
                                  cursor="hand2", width=20,
                                  state="disabled")
        self.stop_btn.pack(pady=(0, 8))

        tk.Button(frame, text="⟳  Refresh All",
                  command=self._refresh_all,
                  bg=ACCENT, fg=BG,
                  font=("Segoe UI", 10, "bold"),
                  relief="flat", padx=14, pady=8,
                  cursor="hand2", width=20).pack(pady=(0, 24))

        steps = [
            "1. Launch Recognition",
            "2. Press [n] to train HC",
            "3. Press [m] to train CNN",
            "4. Press [e] to record test",
            "5. Press [q] to quit",
            "6. Refresh All to update database & CM",
        ]
        tk.Label(frame, text="Workflow", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=16)
        for step in steps:
            tk.Label(frame, text=step, bg=PANEL, fg=SUBTEXT,
                     font=("Segoe UI", 8), anchor="w").pack(fill="x", padx=20, pady=1)

        self.app_status = tk.Label(frame, text="App: not running",
                                   bg=PANEL, fg=SUBTEXT,
                                   font=("Segoe UI", 8, "italic"))
        self.app_status.pack(side="bottom", pady=10)

    def _build_tabs(self, parent):
        right = tk.Frame(parent, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL, foreground=TEXT,
                        padding=[12, 6], font=("Segoe UI", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", BG)])

        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)

        self._build_hc_tab(nb)
        self._build_cnn_tab(nb)
        self._build_cm_tab(nb)

    def _build_hc_tab(self, nb):
        frame = tk.Frame(nb, bg=BG)
        nb.add(frame, text=" Hand-Crafted DB ")

        top = tk.Frame(frame, bg=BG)
        top.pack(fill="x", padx=10, pady=8)

        self.hc_count = tk.StringVar()
        tk.Label(top, textvariable=self.hc_count, bg=BG, fg=TEXT,
                 font=("Segoe UI", 10)).pack(side="left")

        tk.Button(top, text="⟳  Refresh", command=self._refresh_hc,
                  bg=ACCENT, fg=BG, font=("Segoe UI", 9, "bold"),
                  relief="flat", padx=10, cursor="hand2").pack(side="right")
        tk.Button(top, text="🗑  Delete Selected", command=self._delete_hc,
                  bg=RED, fg=BG, font=("Segoe UI", 9, "bold"),
                  relief="flat", padx=10, cursor="hand2").pack(side="right", padx=(0, 6))

        cols = ["Label", "percentFilled", "aspectRatio",
                "hu0", "hu1", "hu2", "hu3", "hu4", "hu5", "hu6"]
        self.hc_tree = self._make_tree(frame, cols)
        self._refresh_hc()

    def _build_cnn_tab(self, nb):
        frame = tk.Frame(nb, bg=BG)
        nb.add(frame, text=" CNN DB ")

        top = tk.Frame(frame, bg=BG)
        top.pack(fill="x", padx=10, pady=8)

        self.cnn_count = tk.StringVar()
        tk.Label(top, textvariable=self.cnn_count, bg=BG, fg=TEXT,
                 font=("Segoe UI", 10)).pack(side="left")

        tk.Button(top, text="⟳  Refresh", command=self._refresh_cnn,
                  bg=ACCENT, fg=BG, font=("Segoe UI", 9, "bold"),
                  relief="flat", padx=10, cursor="hand2").pack(side="right")
        tk.Button(top, text="🗑  Delete Selected", command=self._delete_cnn,
                  bg=RED, fg=BG, font=("Segoe UI", 9, "bold"),
                  relief="flat", padx=10, cursor="hand2").pack(side="right", padx=(0, 6))

        cols = ["Label"] + [f"emb{i}" for i in range(8)] + ["..."]
        self.cnn_tree = self._make_tree(frame, cols)
        self._refresh_cnn()

    def _build_cm_tab(self, nb):
        frame = tk.Frame(nb, bg=BG)
        nb.add(frame, text=" Confusion Matrix ")

        top = tk.Frame(frame, bg=BG)
        top.pack(fill="x", padx=10, pady=8)

        tk.Button(top, text="⟳  Reload", command=self._refresh_cm,
                  bg=ACCENT, fg=BG, font=("Segoe UI", 9, "bold"),
                  relief="flat", padx=10, cursor="hand2").pack(side="right")

        split = tk.Frame(frame, bg=BG)
        split.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.hc_cm_frame  = tk.Frame(split, bg=BG)
        self.cnn_cm_frame = tk.Frame(split, bg=BG)
        self.hc_cm_frame.pack(side="left",  fill="both", expand=True, padx=(0, 6))
        self.cnn_cm_frame.pack(side="left", fill="both", expand=True)

        self._refresh_cm()

    def _make_tree(self, parent, cols):
        style = ttk.Style()
        style.configure("Custom.Treeview",
                        background=PANEL, foreground=TEXT,
                        fieldbackground=PANEL, rowheight=24,
                        borderwidth=0, font=("Segoe UI", 9))
        style.configure("Custom.Treeview.Heading",
                        background=BORDER, foreground=TEXT,
                        font=("Segoe UI", 9, "bold"), relief="flat")
        style.map("Custom.Treeview",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", BG)])

        wrap = tk.Frame(parent, bg=BG)
        wrap.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        tree = ttk.Treeview(wrap, columns=cols, show="headings",
                            style="Custom.Treeview", selectmode="extended")
        for c in cols:
            w = 100 if c == "Label" else 75
            tree.heading(c, text=c)
            tree.column(c, width=w, anchor="center", minwidth=50)

        sb = ttk.Scrollbar(wrap, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True)
        return tree

    def _refresh_hc(self):
        rows = load_csv(HC_DB, HC_DIM)
        self.hc_tree.delete(*self.hc_tree.get_children())
        for r in rows:
            vals = [r[0]] + [f"{float(v):.4f}" for v in r[1:]]
            self.hc_tree.insert("", "end", values=vals)
        self.hc_count.set(f"Entries: {len(rows)}")
        self.status_var.set(f"HC DB loaded — {len(rows)} entries")

    def _refresh_cnn(self):
        rows = load_csv(CNN_DB, CNN_DIM)
        self.cnn_tree.delete(*self.cnn_tree.get_children())
        for r in rows:
            preview = [r[0]] + [f"{float(v):.4f}" for v in r[1:9]] + ["..."]
            self.cnn_tree.insert("", "end", values=preview)
        self.cnn_count.set(f"Entries: {len(rows)}")
        self.status_var.set(f"CNN DB loaded — {len(rows)} entries")

    
    def _delete_hc(self):
        sel = self.hc_tree.selection()
        if not sel:
            messagebox.showwarning("Nothing selected", "Select at least one row to delete.")
            return
        if not messagebox.askyesno("Confirm", f"Delete {len(sel)} selected {'entries' if len(sel) > 1 else 'entry'} from HC DB?"):
            return
        all_items = self.hc_tree.get_children()
        indices_to_delete = {all_items.index(s) for s in sel}
        rows = load_csv(HC_DB, HC_DIM)
        rows = [r for i, r in enumerate(rows) if i not in indices_to_delete]
        save_csv(HC_DB, rows)
        self._refresh_hc()

    def _delete_cnn(self):
        sel = self.cnn_tree.selection()
        if not sel:
            messagebox.showwarning("Nothing selected", "Select at least one row to delete.")
            return
        if not messagebox.askyesno("Confirm", f"Delete {len(sel)} selected {'entries' if len(sel) > 1 else 'entry'} from CNN DB?"):
            return
        all_items = self.cnn_tree.get_children()
        indices_to_delete = {all_items.index(s) for s in sel}
        rows = load_csv(CNN_DB, CNN_DIM)
        rows = [r for i, r in enumerate(rows) if i not in indices_to_delete]
        save_csv(CNN_DB, rows)
        self._refresh_cnn()

    def _refresh_cm(self):
        for w in self.hc_cm_frame.winfo_children():
            w.destroy()
        for w in self.cnn_cm_frame.winfo_children():
            w.destroy()
        self._render_cm(self.hc_cm_frame,  CM_HC,  "Hand-Crafted")
        self._render_cm(self.cnn_cm_frame, CM_CNN, "CNN")

    def _render_cm(self, parent, path, title):
        tk.Label(parent, text=title, bg=BG, fg=ACCENT,
                 font=("Segoe UI", 11, "bold"), pady=6).pack()

        if not os.path.exists(path):
            tk.Label(parent, text="No data yet.\nRun tests with [e] in the C++ app.",
                     bg=BG, fg=SUBTEXT, font=("Segoe UI", 9)).pack(pady=20)
            return

        labels, matrix, accuracies = self._parse_cm_file(path)
        if not labels:
            tk.Label(parent, text="Could not parse file.",
                     bg=BG, fg=RED, font=("Segoe UI", 9)).pack(pady=20)
            return

        grid = tk.Frame(parent, bg=BG)
        grid.pack(padx=6, pady=4)

        tk.Label(grid, text="", bg=BG, width=12).grid(row=0, column=0)
        for ci, lbl in enumerate(labels):
            tk.Label(grid, text=lbl, bg=BG, fg=SUBTEXT,
                     font=("Segoe UI", 8, "bold"), width=10,
                     anchor="center").grid(row=0, column=ci + 1)

        for ri, rlbl in enumerate(labels):
            tk.Label(grid, text=rlbl, bg=BG, fg=TEXT,
                     font=("Segoe UI", 8, "bold"), anchor="e",
                     width=12).grid(row=ri + 1, column=0, padx=(0, 4))
            for ci, val in enumerate(matrix[ri]):
                if ri == ci:
                    bg = GREEN if val > 0 else PANEL
                    fg = BG    if val > 0 else SUBTEXT
                else:
                    bg = RED   if val > 0 else PANEL
                    fg = BG    if val > 0 else SUBTEXT
                tk.Label(grid, text=str(val), bg=bg, fg=fg,
                         font=("Segoe UI", 9, "bold"),
                         width=5, padx=4, pady=4).grid(row=ri + 1, column=ci + 1,
                                                        padx=2, pady=2)

        tk.Label(parent, text="Per-class accuracy",
                 bg=BG, fg=SUBTEXT,
                 font=("Segoe UI", 8, "bold"), pady=6).pack()

        for lbl, acc in accuracies:
            color = GREEN if acc >= 66 else (YELLOW if acc >= 33 else RED)
            row = tk.Frame(parent, bg=BG)
            row.pack(fill="x", padx=16, pady=1)
            tk.Label(row, text=lbl, bg=BG, fg=TEXT,
                     font=("Segoe UI", 8), width=14, anchor="w").pack(side="left")
            tk.Label(row, text=f"{acc:.1f}%", bg=BG, fg=color,
                     font=("Segoe UI", 8, "bold")).pack(side="left")

    def _parse_cm_file(self, path):
        with open(path) as f:
            lines = f.readlines()

        labels, matrix, accuracies = [], [], []

        header_idx = None
        for i, line in enumerate(lines):
            if "<- Predicted" in line:
                header_idx = i
                break
        if header_idx is None:
            return [], [], []

        labels = lines[header_idx].replace("<- Predicted", "").split()

        for line in lines[header_idx + 2:]:
            if not line.strip() or line.startswith("Rows"):
                break
            nums = []
            for p in line.split():
                clean = p.replace("*", "")
                if clean.lstrip("-").isdigit():
                    nums.append(int(clean))
            if nums:
                matrix.append(nums)

        in_acc = False
        for line in lines:
            if "Per-class accuracy" in line:
                in_acc = True
                continue
            if in_acc and ":" in line and "%" in line:
                lbl = line.strip().split(":")[0].strip()
                try:
                    acc = float(line[line.rfind("(") + 1: line.rfind("%")])
                    accuracies.append((lbl, acc))
                except ValueError:
                    pass

        return labels, matrix, accuracies

    # C++ process management
    def _launch_cpp(self):
        exe = PROJECT_EXE
        if not os.path.exists(exe):
            messagebox.showerror("Not found", f"Could not find {exe} in the current folder.")
            return

        self.launch_btn.config(state="disabled", text="Running...")
        self.stop_btn.config(state="normal")
        self.app_status.config(text="App: running", fg=GREEN)

        self._proc = subprocess.Popen(
            [os.path.abspath(exe)],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            cwd=os.getcwd()
        )

        def _watch():
            self._proc.wait()                      # block until process exits
            self.launch_btn.config(state="normal", text="▶  Launch Recognition")
            self.stop_btn.config(state="disabled")
            self.app_status.config(text="App: finished", fg=YELLOW)
            self._refresh_all()

        threading.Thread(target=_watch, daemon=True).start()

    def _stop_cpp(self):
        self._kill_proc()
        self.launch_btn.config(state="normal", text="▶  Launch Recognition")
        self.stop_btn.config(state="disabled")
        self.app_status.config(text="App: stopped", fg=RED)
        self._refresh_all()

    def _refresh_all(self):
        self._refresh_hc()
        self._refresh_cnn()
        self._refresh_cm()
        self.status_var.set("Refreshed all")


if __name__ == "__main__":
    app = App()
    app.mainloop()
# Arpit Gandhi
# April 2026
# This file defines testing for trained models on MNIST and custom handwritten digit images,
# as well as transfer learning to classify greek letters

import os
import matplotlib.pyplot as plt
import torch
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from modelTrain import MyNetwork, BATCH_SIZE_TEST, GaborNetwork
from NetTransformer import NetTransformer, NetConfig

LEARNING_RATE = 0.01
MOMENTUM = 0.5
N_EPOCHS = 50

# load MNIST data
def load_data(data_path):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(data_path, train=False, download=False, transform=transform),
        batch_size=BATCH_SIZE_TEST, shuffle=False)

# load custom handwritten digit images
def load_handwritten_digits(images_path):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    images = []
    for i in range(10):
        img = Image.open(os.path.join(images_path, f'{i}.jpg'))
        images.append(transform(img))
    return torch.stack(images)

# print predictions for MNIST test examples
def print_predictions(output, targets):
    for i in range(10):
        values = ['{:.2f}'.format(v) for v in output[i].tolist()]
        print(f'Example {i+1}: [{", ".join(values)}]')
        print(f'         Predicted: {output[i].argmax().item()}, True Label: {targets[i].item()}\n')

# print predictions for handwritten digits
def print_handwritten_predictions(output):
    for i in range(10):
        values = ['{:.2f}'.format(v) for v in output[i].tolist()]
        print(f'Digit {i}: [{", ".join(values)}]')
        print(f'         Predicted: {output[i].argmax().item()}, True Label: {i}\n')

# visualize filters and their effects
def plot_examples(example_data, output, title):
    fig = plt.figure()
    fig.suptitle(title, fontsize=12)
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([]); plt.yticks([])
    plt.show()

# visualize filters and their effects
def plot_handwritten_digits(images, output, title):
    fig = plt.figure()
    fig.suptitle(title, fontsize=12)
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.tight_layout()
        plt.imshow(images[i][0], cmap='gray', interpolation='none')
        plt.title(f'Pred: {output[i].argmax().item()}\nTrue: {i}')
        plt.xticks([]); plt.yticks([])
    plt.show()


# Greek Transfer Learning
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)

# load greek letter training data
def load_greek_data(greek_path):
    return torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            greek_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                GreekTransform(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ])),
        batch_size=5, shuffle=True)

# prepare model for transfer learning
def prepare_model(model_path):
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    for param in model.parameters():
        param.requires_grad = False
    model.fc2 = nn.Linear(50, 3)
    return model

# train on greek letters
def train_greek(model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss+= loss.item()
        correct+= output.argmax(dim=1).eq(target).sum().item()
        total+= len(target)
    mean_loss = total_loss/len(train_loader)
    accuracy = 100.0*correct/total
    print(f"Epoch {epoch:3d} | loss: {mean_loss:.4f} | accuracy: ({accuracy:.1f}%)")
    return mean_loss, accuracy

# visualize training progress
def plot_greek_training(losses, accuracies, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(range(1, len(losses)+1), losses, color="tab:blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean Training Loss")
    ax1.set_title("Training Loss"); ax1.grid(True)

    ax2.plot(range(1, len(accuracies)+1), accuracies, color="tab:orange")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training Accuracy")
    ax2.set_ylim(0, 105)
    ax2.axhline(100, linestyle="--", color="gray", linewidth=0.8); ax2.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

# evaluate on custom test set of greek letters
def evaluate_custom(model, custom_path, device):
    CLASS_NAMES = ["alpha", "beta", "gamma"]
    dataset = torchvision.datasets.ImageFolder(custom_path,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), GreekTransform(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    correct = 0
    print(f"\n{'TestImg':<15} {'True':>6}  {'Pred':>6}")
    for idx, (data, target) in enumerate(loader):
        data = data.to(device) 
        pred = model(data).argmax(dim=1).item()
        true = target.item()
        fname = os.path.basename(dataset.samples[idx][0])
        correct+= int(pred == true)
        print(f"{fname:<15} {CLASS_NAMES[true]:>6}  {CLASS_NAMES[pred]:>6}")
    print(f"\nOverall: ({100*correct/len(dataset):.1f}%)")


def main():
    base = os.path.dirname(__file__)
    data_path = os.path.join(base, 'data')
    digits_path = os.path.join(base, 'handwritten_digits')
    greek_path = os.path.join(base, 'greek_train')
    custom_path = os.path.join(base, 'greek_test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_loader = load_data(data_path)
    _, (example_data, example_targets) = next(enumerate(test_loader))
    handwritten_images = load_handwritten_digits(digits_path)

    # Test models
    test_models = {
        "CNN": (os.path.join(base, 'models', 'model.pth'), MyNetwork()),
        "GaborCNN": (os.path.join(base, 'models', 'gabor_model.pth'), GaborNetwork()),
        "Transformer": (os.path.join(base, 'models', 'transformer_model.pth'), NetTransformer(NetConfig())),
    }

    for name, (model_path, model) in test_models.items():
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()
        print(f"{summary(model, (1, 1, 28, 28))}")

        with torch.no_grad():
            output = model(example_data[:10].to(device))
            handwritten_output = model(handwritten_images.to(device))

        print(f'{name}: MNIST Test Examples')
        print_predictions(output, example_targets)
        plot_examples(example_data, output, title=f'{name}: MNIST Test Examples')

        print(f'{name}: Handwritten Digits')
        print_handwritten_predictions(handwritten_output)
        plot_handwritten_digits(handwritten_images, handwritten_output, title=f'{name}: Handwritten Digits')

    # Greek transfer learning
    cnn_model_path = os.path.join(base, 'models', 'model.pth')
    greek_model = prepare_model(cnn_model_path)
    greek_model = greek_model.to(device)
    print(f"{summary(greek_model, (1, 1, 28, 28))}")

    train_loader = load_greek_data(greek_path)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, greek_model.parameters()),
                          lr=LEARNING_RATE, momentum=MOMENTUM)
    criterion = nn.CrossEntropyLoss()

    losses, accuracies = [], []
    for epoch in range(1, N_EPOCHS+1):
        loss, acc = train_greek(greek_model, train_loader, optimizer, criterion, epoch, device)
        losses.append(loss)
        accuracies.append(acc)

    plot_greek_training(losses, accuracies)
    torch.save(greek_model.state_dict(), os.path.join(base, 'models', 'greek_model.pth'))

    if os.path.isdir(custom_path):
        evaluate_custom(greek_model, custom_path, device)


if __name__ == "__main__":
    main()
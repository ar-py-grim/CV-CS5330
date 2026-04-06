# Arpit Gandhi
# April 2026
# This file defines the CNN architectures and training functions for the MNIST digit classification task

import os
import numpy as np
import glob
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from NetTransformer import NetTransformer, NetConfig

TRAIN_TRANSFORMER = False
TRAIN_GABOR = False

# Hyperparameters
N_EPOCHS = 10
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000
LEARNING_RATE = 0.01
MOMENTUM = 0.5
LOG_INTERVAL = 10
RANDOM_SEED = 1

# Gabor filter bank
def make_gabor_bank(n_filters=10, size=5):
    filters = []
    for lam in [4.0, 7.0]:
        for i in range(n_filters//2):
            theta = i*np.pi/(n_filters//2)
            half = size//2
            y, x = np.mgrid[-half:half+1, -half:half+1]
            x_t = x*np.cos(theta) + y*np.sin(theta)
            y_t = -x*np.sin(theta) + y*np.cos(theta)
            k = np.exp(-(x_t**2 + 0.25*y_t**2)/8)*np.cos(2*np.pi*x_t/lam)
            k -= k.mean()
            filters.append(k)
    bank = np.stack(filters[:n_filters])[:, np.newaxis]  # (10,1,5,5)
    return torch.tensor(bank, dtype=torch.float32)

# Define CNN with optional Gabor filters
class GaborNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, bias=False)
        with torch.no_grad():
            self.conv1.weight.copy_(make_gabor_bank())
        self.conv1.weight.requires_grad_(False)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

# Standard CNN for comparison
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), kernel_size=2, stride=2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# load MNIST test, train data
def load_data(data_path):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(data_path, train=True, download=True, transform=transform),
        batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(data_path, train=False, download=True, transform=transform),
        batch_size=BATCH_SIZE_TEST, shuffle=True)
    
    # remove raw MNIST files
    for gz_file in glob.glob(os.path.join(data_path, 'MNIST', 'raw', '*.gz')):
        os.remove(gz_file)

    return train_loader, test_loader

# plot some MNIST examples
def plot_examples(test_loader):
    examples = enumerate(test_loader)
    _, (example_data, example_targets) = next(examples)
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("True Label: {}".format(example_targets[i]))
        plt.xticks([]); plt.yticks([])
    plt.show()

# train functions
def train(epoch, network, optimizer, train_loader, train_losses, train_counter, results_path, model_name, device):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loader.dataset),
                100.*batch_idx/len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), os.path.join(results_path, model_name))
            torch.save(optimizer.state_dict(), os.path.join(results_path, 'optimizer.pth'))

# test function
def test(network, test_loader, test_losses, device):
    network.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss+= F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct+= pred.eq(target.data.view_as(pred)).sum()
    test_loss/= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100.*correct/len(test_loader.dataset)))


def main():
    torch.backends.cudnn.enabled = False
    torch.manual_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base = os.path.dirname(__file__)
    data_path = os.path.join(base, 'data')
    results_path = os.path.join(base, 'models')
    os.makedirs(results_path, exist_ok=True)

    train_loader, test_loader = load_data(data_path)
    plot_examples(test_loader)

    if TRAIN_TRANSFORMER:
        config = NetConfig()
        network = NetTransformer(config)
        optimizer = optim.AdamW(network.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        model_name = 'transformer_model.pth'
    
    elif TRAIN_GABOR:
        network = GaborNetwork()
        network = network.to(device)
        trainable = [p for p in network.parameters() if p.requires_grad]
        optimizer = optim.SGD(trainable, lr=LEARNING_RATE, momentum=MOMENTUM)
        model_name = 'gabor_model.pth'

    else:
        network = MyNetwork()
        network = network.to(device)
        optimizer = optim.SGD(network.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
        model_name = 'model.pth'

    print(network)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(1, N_EPOCHS+1)]

    for epoch in range(1, N_EPOCHS+1):
        train(epoch, network, optimizer, train_loader, train_losses, train_counter,
             results_path, model_name, device)
        test(network, test_loader, test_losses, device)

    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


if __name__ == "__main__":
    main()
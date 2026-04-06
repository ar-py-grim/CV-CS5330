# Arpit Gandhi
# April 2026
# This file defines visualization functions for trained CNN models on MNIST,
# as well as visualizing filters from a pretrained ResNet18 model. 
# It includes plotting filter weights, their effects on input images, and comparing with ResNet filters.

import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import models
import cv2
from modelTrain import MyNetwork, BATCH_SIZE_TRAIN

def load_data(data_path):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(data_path, train=True, download=False, transform=transform),
        batch_size=BATCH_SIZE_TRAIN, shuffle=True)

def print_filter_weights(weights):
    print(f'Filter weights shape: {weights.shape}')
    for i in range(weights.shape[0]):
        print(f'Filter {i}:\n{weights[i, 0]}')

def plot_filters(weights, title='Model Conv1 Weights'):
    n = weights.shape[0]
    plt.figure()
    plt.suptitle(title)
    for i in range(n):
        plt.subplot(3, 4, i+1)
        plt.tight_layout()
        plt.imshow(weights[i, 0].detach().numpy(), cmap='viridis', interpolation='none')
        plt.title(f'Filter {i}')
        plt.xticks([]); plt.yticks([])
    plt.show()

def plot_filtered_images(weights, example_data, title='Filter effects'):
    n = weights.shape[0]
    img = example_data[0, 0].detach().numpy()
    plt.figure()
    plt.suptitle(title, fontsize=11)
    for i in range(n):
        kernel   = weights[i, 0].detach().numpy()
        filtered = cv2.filter2D(img, -1, kernel)

        plt.subplot(5, 4, i*2+1)
        plt.tight_layout()
        plt.imshow(kernel, cmap='gray', interpolation='none')
        plt.xticks([]); plt.yticks([])
        plt.subplot(5, 4, i*2+2)
        plt.tight_layout()
        plt.imshow(filtered, cmap='gray', interpolation='none')
        plt.xticks([]); plt.yticks([])
    plt.show()

def plot_res_filters(layer, title):
    """Visualize all filters from a ResNet conv layer."""
    weights = layer.weight.detach().cpu()
    n, cols = weights.shape[0], 8
    rows = (n+cols-1)//cols
    plt.figure(figsize=(cols*2, rows*2))
    plt.suptitle(title, fontsize=11)
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        plt.imshow(weights[i].mean(0).numpy(), cmap='viridis', interpolation='none')
        plt.title(f'{i}', fontsize=7)
        plt.xticks([]); plt.yticks([])
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.4)
    plt.show()

def resnet_visualise(example_data):
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.eval()
    print(net)
    # (64, 3, 7, 7)
    conv1 = net.conv1     
    # (64, 64, 3, 3)       
    conv2 = net.layer1[0].conv1  
    print(f'Conv1 weight shape: {conv1.weight.shape}')
    print(f'Conv2 weight shape: {conv2.weight.shape}')

    # Visualize filter kernels
    plot_res_filters(conv1, 'ResNet18 Conv1 (64 filters, 7x7)')
    plot_res_filters(conv2, 'ResNet18 Conv2 (64 filters, 3x3, avg over 64 in-channels)')

    # Visualize filter effects on a digit (avg input channels-> grayscale kernel)
    w1_gray = conv1.weight.detach().mean(dim=1, keepdim=True)[:10]
    w2_gray = conv2.weight.detach().mean(dim=1, keepdim=True)[:10]
    plot_filtered_images(w1_gray, example_data, title='ResNet18 Conv1 filter effect on digit')
    plot_filtered_images(w2_gray, example_data, title='ResNet18 Conv2 filter effect on digit')


def main():
    base = os.path.dirname(__file__)
    data_path = os.path.join(base, 'data')
    model_path = os.path.join(base, 'models', 'model.pth')

    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(model)

    train_loader = load_data(data_path)
    _, (example_data, _) = next(enumerate(train_loader))
    weights = model.conv1.weight

    # MNIST network filters
    print_filter_weights(weights)
    plot_filters(weights)
    plot_filtered_images(weights, example_data, title='Model Conv1 filter effect on digit')

    # ResNet18 filters
    resnet_visualise(example_data)


if __name__ == "__main__":
    main()
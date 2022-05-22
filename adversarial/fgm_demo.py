import numpy as np
import torch
import torch.nn as nn
import torchvision

from torch.utils.data import DataLoader
from functools import partial
from fgm import fgm
from mnist_st_demo import Basic_CNN, get_accuracy, train, plot_adversarial


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Basic_CNN(1, 10).to(device)
    train_data = torchvision.datasets.MNIST("./", train=True, transform=torchvision.transforms.ToTensor())
    test_data = torchvision.datasets.MNIST("./", train=False, transform=torchvision.transforms.ToTensor())
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)
    loss_fun = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)

    train(model, train_dataloader, loss_fun, optimizer, epoch=3, device=device)
    print("[INFO] Origin", get_accuracy(model, test_dataloader, device=device))

    # attack = partial(fgm, eps=1, norm=np.inf, targeted=True)
    # print("[INFO] FGM", get_accuracy(model, test_dataloader, attack, device=device))
    # plot_adversarial(model, test_dataloader, attack, device=device, number=3)

    attack = partial(fgm, eps=0.1, norm=np.inf, targeted=False)
    print("[INFO] FGM", get_accuracy(model, test_dataloader, attack, device=device))
    plot_adversarial(model, test_dataloader, attack, device=device, number=3)

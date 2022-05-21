import torch
import numpy as np
import torch.nn as nn
import torchvision
from functools import partial
from torch.utils.data import DataLoader
from cleverhans.torch.utils import optimize_linear
from adversarial.mnist_demo import Basic_CNN, get_accuracy, train, plot_adversarial


def fgm(model_fn, x, eps, norm, loss_fn=None,
        clip_min=None, clip_max=None, y=None, targeted=False):
    x = x.clone().detach().to(torch.float).requires_grad_(True)
    if y is None:
        _, y = torch.max(model_fn(x), 1)

    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(model_fn(x), y)

    if targeted:
        loss = -loss

    loss.backward()
    optimal_perturbation = optimize_linear(x.grad, eps, norm)
    adv_x = x + optimal_perturbation
    if (clip_min is not None) or (clip_max is not None):
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    return adv_x


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

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import tqdm
from functools import partial


class Basic_CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Basic_CNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv1_1 = nn.Conv2d(self.in_channels, 32, kernel_size=(3, 3), padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 200)
        self.fc2 = nn.Linear(200, self.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(model, train_dataloader, loss_fun, optimizer, epoch=1, device="cuda"):
    model.train()
    for i in range(epoch):
        for (x, y) in tqdm.tqdm(train_dataloader, desc=f"[INFO] Epoch {i + 1}"):
            optimizer.zero_grad()
            predict = model(x.to(device))
            losses = loss_fun(predict, y.to(device))
            losses.backward()
            optimizer.step()


def get_accuracy(model, data_loader, attack=None, device="cuda"):
    model.eval()
    acc_count = 0
    num_count = 0
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        if attack:
            x = attack(model, x)
        out = model(x)
        acc_count += torch.eq(y, torch.argmax(out, 1)).sum()
        num_count += y.shape[0]

    return (acc_count / num_count).cpu().numpy()


def plot_adversarial(model, data_loader, attack=None, device="cuda", number=10):
    model.eval()
    i = 0
    for x, y in data_loader:
        x = x.to(device)
        y = y.cpu().numpy()
        assert attack, "[ERROR] In 'plot_adversarial' attack is None."
        x = attack(model, x)
        predict = model(x).argmax(dim=1).detach().cpu().numpy()
        for x_, y_, predict_ in zip(x, y, predict):
            i += 1
            plt.imshow(x_[0].detach().cpu().numpy(), cmap='gray')
            plt.title(f"label: {y_} predict: {predict_}")
            plt.show()

            if i >= number:
                break

        if i >= number:
            break


if __name__ == '__main__':
    from fgsm import fgsm
    from fgm import fgm
    from pgd import pgd

    # from lipschitz import local_lipschitz

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

    attack = partial(fgsm, eps=28)
    print("[INFO] FGSM", get_accuracy(model, test_dataloader, attack, device=device))
    plot_adversarial(model, test_dataloader, attack, device=device, number=1)

    attack = partial(fgm, eps=0.1, norm=np.inf)
    print("[INFO] FGM", get_accuracy(model, test_dataloader, attack, device=device))
    plot_adversarial(model, test_dataloader, attack, device=device, number=1)

    attack = partial(pgd, eps=0.1, norm=np.inf, eps_iter=10, nb_iter=10)
    print("[INFO] PGD", get_accuracy(model, test_dataloader, attack, device=device))
    plot_adversarial(model, test_dataloader, attack, device=device, number=1)

    # attack = partial(local_lipschitz, top_norm=1, btm_norm=np.Inf, step_size=0.3, epsilon=10)
    # print("[INFO] FGSM", get_accuracy(model, test_dataloader, attack, device=device))
    # attack = partial(local_lipschitz, top_norm=1, btm_norm=np.Inf, step_size=0.3, epsilon=10, print_eta = False)
    # plot_adversarial(model, test_dataloader, attack, device=device, number=10)

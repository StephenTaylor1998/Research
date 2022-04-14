import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.utils import save_image


class Encoder(nn.Module):
    def __init__(self, channel, lantent_size, num_class):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(channel, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mu = nn.Linear(128 + num_class, lantent_size)
        self.log_var = nn.Linear(128 + num_class, lantent_size)

    def forward(self, x, condition):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = self.pool(x)
        x = torch.cat([x.squeeze(-1).squeeze(-1), condition], dim=1)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, channel, lantent_size, num_class):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear((lantent_size + num_class), 128 * 7 * 7)
        self.tconv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.tconv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.tconv3 = nn.Conv2d(32, channel, kernel_size=3, padding=1)

    def forward(self, lantent, condition):
        x = torch.cat([lantent, condition], dim=1)
        x = torch.tanh(self.linear1(x).reshape(-1, 128, 7, 7))
        x = torch.tanh(F.interpolate(self.tconv1(x), scale_factor=2))
        x = torch.tanh(F.interpolate(self.tconv2(x), scale_factor=2))
        reconstruction = torch.sigmoid(self.tconv3(x))
        return reconstruction


class Sampler(nn.Module):
    def forward(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        lantent = eps.mul(std).add(mu)
        return lantent


class CVAE(nn.Module):
    def __init__(self, channel, lantent_size, num_class):
        super(CVAE, self).__init__()
        self.encoder = Encoder(channel, lantent_size, num_class)
        self.sampler = Sampler()
        self.decoder = Decoder(channel, lantent_size, num_class)

    def forward(self, inputs, condition):
        mu, log_var = self.encoder(inputs, condition)
        lantent = self.sampler(mu, log_var)
        outputs = self.decoder(lantent, condition)
        return outputs, mu, log_var


def lr_scheduler(optimizer, epoch, lr):
    if epoch < 2:
        lr = lr * 0.01
    elif epoch < 20:
        lr = lr
    elif epoch < 30:
        lr = lr * 0.1
    elif epoch < 40:
        lr = lr * 0.05
    else:
        lr = lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


device = "cuda" if torch.cuda.is_available() else "cpu"

bs = 512
lr = 1e-3
train_dataset = datasets.FashionMNIST(root='./', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.FashionMNIST(root='./', train=False, transform=transforms.ToTensor(), download=False)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, num_workers=6)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False, num_workers=6)

cond_dim = train_loader.dataset.targets.unique().size(0)
cvae = CVAE(1, 2, 10).to(device)
optimizer = optim.Adam(cvae.parameters(), lr)


def loss_function(reconstruction_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(reconstruction_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def one_hot(array, num_classes):
    return torch.eye(num_classes, dtype=torch.int)[array]


def train_cvae(epoch):
    cvae.train()
    train_loss = 0
    for batch_idx, (data, cond) in enumerate(train_loader):
        optimizer.zero_grad()
        data, cond = data.to(device), one_hot(cond, cond_dim).to(device)
        reconstruction_batch, mu, log_var = cvae(data, cond)
        loss = loss_function(reconstruction_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Batch Index {}\tLoss: {:.4f}'.format(
                batch_idx , loss.item() / len(data)))
    print('[INFO] Train Epoch: {} loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def test_cvae():
    cvae.eval()
    test_loss = 0
    with torch.no_grad():
        for data, cond in test_loader:
            data, cond = data.to(device), one_hot(cond, cond_dim).to(device)
            reconstruction, mu, log_var = cvae(data, cond)
            test_loss += loss_function(reconstruction, data, mu, log_var).item()

    test_loss /= len(test_loader.dataset)
    print('[INFO] Test loss: {:.4f}'.format(test_loss))


if __name__ == '__main__':

    for epoch in range(1, 51):
        lr_scheduler(optimizer, epoch, lr)
        train_cvae(epoch)
        test_cvae()

    with torch.no_grad():
        z = torch.randn(10, 2).to(device)
        c = torch.eye(10).to(device)
        sample = cvae.decoder(z, c)
        save_image(sample.view(10, 1, 28, 28), './samples/sample_' + '.png')

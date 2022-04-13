import torch
from linear import Linear


if __name__ == '__main__':
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.float)
    target = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.float)
    linear = Linear(8, 8, bias=True)
    optimizer = torch.optim.SGD(linear.parameters(), 0.01)
    loss_fun = torch.nn.MSELoss()

    for i in range(10):
        linear.zero_grad()
        output = linear(inputs)
        loss = loss_fun(output, target)
        print(loss.item())
        loss.backward()
        optimizer.step()

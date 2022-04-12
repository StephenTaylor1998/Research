import numpy as np


def fun(weight, bias, x):
    return 1 if weight @ x + bias >= 0 else -1


def pla(weight, bias, x_y, epoch=10000):
    np.random.shuffle(x_y)
    for t in range(epoch):
        flag = True
        for (x, y) in x_y:
            x, y = np.array(x), np.array(y)
            y_hat = fun(weight, bias, x)
            if y_hat != y:
                weight = weight + (x * y)
                bias = bias + y
                flag = False

        if flag: break

    return weight, bias

if __name__ == '__main__':
    weight = np.array([0, 0])
    bias = np.array([0])
    x_y = [
        [[1, 1], -1],
        [[3, 3], 1],
        [[4, 3], 1],

    ]
    weight, bias = pla(weight, bias, x_y)
    print("weight:", weight, "bias:", bias)

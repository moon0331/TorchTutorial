import math
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

import mnist

def get_w_b():
    weights = torch.randn(784, 10) / math.sqrt(784)
    weights.requires_grad_()
    bias = torch.zeros(10, requires_grad=True)

    return weights, bias


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias


class Mnist_Logistic_Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


def simple_model(bs):
    (x_train, y_train), (x_valid, y_valid) = mnist._load_mnist()
    weights, bias = get_w_b()

    def log_softmax(x):
        return x - x.exp().sum(-1).log().unsqueeze(-1)

    def model(xb):
        return log_softmax(xb @ weights + bias)

    def nll(input, target):
        return -input[range(target.shape[0]), target].mean()

    loss_func = nll

    lr = 0.5  # 학습률(learning rate)
    epochs = 2  # 훈련에 사용할 에폭(epoch) 수
    n, c = x_train.shape

    xb0 = x_train[:bs]
    yb0 = y_train[:bs]

    print(loss_func(model(x_train), y_train), accuracy(model(x_train), y_train))

    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            #         set_trace()
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                weights -= weights.grad * lr
                bias -= bias.grad * lr
                weights.grad.zero_()
                bias.grad.zero_()

    print(loss_func(model(x_train), y_train), accuracy(model(x_train), y_train))


def functional_model(bs):
    (x_train, y_train), (x_valid, y_valid) = mnist._load_mnist()

    loss_func = F.cross_entropy

    weights, bias = get_w_b()

    def model(xb):
        return xb @ weights + bias

    lr = 0.5  # 학습률(learning rate)
    epochs = 2  # 훈련에 사용할 에폭(epoch) 수
    n, c = x_train.shape

    xb0 = x_train[:bs]
    yb0 = y_train[:bs]

    print(loss_func(model(x_train), y_train), accuracy(model(x_train), y_train))

    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            #         set_trace()
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                weights -= weights.grad * lr
                bias -= bias.grad * lr
                weights.grad.zero_()
                bias.grad.zero_()

    print(loss_func(model(x_train), y_train), accuracy(model(x_train), y_train))


def nn_module_model(bs, linear=False):
    (x_train, y_train), (x_valid, y_valid) = mnist._load_mnist()

    loss_func = F.cross_entropy

    if linear:
        model = Mnist_Logistic_Linear()
    else:
        model = Mnist_Logistic()

    lr = 0.5  # 학습률(learning rate)
    epochs = 2  # 훈련에 사용할 에폭(epoch) 수
    n, c = x_train.shape

    xb0 = x_train[:bs]
    yb0 = y_train[:bs]

    print(loss_func(model(x_train), y_train), accuracy(model(x_train), y_train))

    # make function fit()
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters(): 
                    p -= p.grad * lr
                model.zero_grad()

    print(loss_func(model(x_train), y_train), accuracy(model(x_train), y_train))


def get_model(lr):
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)


def nn_module_optim_model(bs):
    (x_train, y_train), (x_valid, y_valid) = mnist._load_mnist()

    loss_func = F.cross_entropy

    lr = 0.5  # 학습률(learning rate)
    epochs = 2  # 훈련에 사용할 에폭(epoch) 수
    n, c = x_train.shape

    model, opt = get_model(lr)

    xb0 = x_train[:bs]
    yb0 = y_train[:bs]

    print(loss_func(model(x_train), y_train), accuracy(model(x_train), y_train))

    # make function fit()
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            opt.step()
            opt.zero_grad()

    print(loss_func(model(x_train), y_train), accuracy(model(x_train), y_train))


def nn_module_optim_dataset_model(bs, data_loader=False):
    (x_train, y_train), (x_valid, y_valid) = mnist._load_mnist()
    train_ds = TensorDataset(x_train, y_train)
    if data_loader:
        train_dl = DataLoader(train_ds, bs)

    loss_func = F.cross_entropy

    lr = 0.5  # 학습률(learning rate)
    epochs = 2  # 훈련에 사용할 에폭(epoch) 수
    n, c = x_train.shape

    model, opt = get_model(lr)

    xb0 = x_train[:bs]
    yb0 = y_train[:bs]

    print(loss_func(model(x_train), y_train), accuracy(model(x_train), y_train))

    # make function fit()
    if data_loader:
        for epoch in range(epochs):
            for xb, yb in train_dl:
                pred = model(xb)
                loss = loss_func(pred, yb)

                loss.backward()
                opt.step()
                opt.zero_grad()
    else:
        for epoch in range(epochs):
            for i in range((n - 1) // bs + 1):
                start_i = i * bs
                end_i = start_i + bs
                xb, yb = train_ds[start_i:end_i]
                pred = model(xb)
                loss = loss_func(pred, yb)

                loss.backward()
                opt.step()
                opt.zero_grad()

    print(loss_func(model(x_train), y_train), accuracy(model(x_train), y_train))


def nn_module_optim_dataset_valid_model(bs):
    (x_train, y_train), (x_valid, y_valid) = mnist._load_mnist()
    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, bs)
    valid_ds = TensorDataset(x_valid, y_valid)
    valid_dl = DataLoader(valid_ds, bs*2)

    loss_func = F.cross_entropy

    lr = 0.5  # 학습률(learning rate)
    epochs = 2  # 훈련에 사용할 에폭(epoch) 수
    n, c = x_train.shape

    model, opt = get_model(lr)

    xb0 = x_train[:bs]
    yb0 = y_train[:bs]

    print(loss_func(model(x_train), y_train), accuracy(model(x_train), y_train))

    # make function fit()
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        with torch.no_grad():
            valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

        print(epoch, valid_loss / len(valid_dl))


    print(loss_func(model(x_train), y_train), accuracy(model(x_train), y_train))


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


import numpy as np

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


def nn_module_optim_dataset_fit_get_data(bs):
    (x_train, y_train), (x_valid, y_valid) = mnist._load_mnist()
    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)

    loss_func = F.cross_entropy

    lr = 0.5  # 학습률(learning rate)
    epochs = 2  # 훈련에 사용할 에폭(epoch) 수
    n, c = x_train.shape

    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
    model, opt = get_model(lr)
    fit(epochs, model, loss_func, opt, train_dl, valid_dl)


class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))


def cnn_module(bs):
    (x_train, y_train), (x_valid, y_valid) = mnist._load_mnist()
    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)

    loss_func = F.cross_entropy

    epochs = 2  # 훈련에 사용할 에폭(epoch) 수
    n, c = x_train.shape

    lr = 0.1
    model = Mnist_CNN()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
    fit(epochs, model, loss_func, opt, train_dl, valid_dl)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

# check above this
def preprocess(x, y=None):
    if y is None:
        # print('preprocess 1')
        return x.view(-1, 1, 28, 28)
    else:
        # print('preprocess 2')
        return x.view(-1, 1, 28, 28), y


def cnn_sequential(bs):
    (x_train, y_train), (x_valid, y_valid) = mnist._load_mnist()
    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)

    loss_func = F.cross_entropy

    epochs = 2  # 훈련에 사용할 에폭(epoch) 수
    n, c = x_train.shape

    lr = 0.1
    model = nn.Sequential(
        Lambda(preprocess),
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(4),
        Lambda(lambda x: x.view(x.size(0), -1)),
    )
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
    fit(epochs, model, loss_func, opt, train_dl, valid_dl)


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

#fancy way?
def preprocess_dev(x, y):
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)


def cnn_sequential_adaptive(bs):
    (x_train, y_train), (x_valid, y_valid) = mnist._load_mnist()
    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)

    loss_func = F.cross_entropy

    epochs = 2  # 훈련에 사용할 에폭(epoch) 수
    n, c = x_train.shape

    lr = 0.1
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), ##
        Lambda(lambda x: x.view(x.size(0), -1)),
    )
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
    train_dl = WrappedDataLoader(train_dl, preprocess_dev)
    valid_dl = WrappedDataLoader(train_dl, preprocess_dev)
    fit(epochs, model, loss_func, opt, train_dl, valid_dl)


if __name__ == '__main__':
    # simple_model(64)
    # functional_model(64)
    # nn_module_model(64)
    # nn_module_model(64, linear=True)
    # nn_module_optim_model(64)
    # nn_module_optim_dataset_model(64)
    # nn_module_optim_dataset_model(64, data_loader=True)
    # nn_module_optim_dataset_valid_model(64)
    # print('after this, not print whole loss : ')
    # nn_module_optim_dataset_fit_get_data(64)
    # cnn_module(64)
    # cnn_sequential(64)
    cnn_sequential_adaptive(64)
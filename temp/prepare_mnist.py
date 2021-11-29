from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

from matplotlib import pyplot
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
) # tensor로 만들기

n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()

print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())

import math

# weights = torch.randn(784, 10) / math.sqrt(784)
# weights.requires_grad_() #randn으로 생성된 텐서는 false가 기본값
# bias = torch.zeros(10, requires_grad=True)

# def log_softmax(x):
#     return x - x.exp().sum(-1).log().unsqueeze(-1) 
#     #https://discuss.pytorch.org/t/log-softmax-function-in-pytorch-tutorial-example/52041

# def model(xb):
#     return log_softmax(xb @ weights + bias)
#     #https://stackoverflow.com/questions/44524901/how-to-do-product-of-matrices-in-pytorch

# def model(xb):
#     return xb @ weights + bias

lr = 0.5  # 학습률(learning rate)
epochs = 10  # 훈련에 사용할 에폭(epoch) 수


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        # self.weights = nn.Parameter(torch.randn(784,10) / math.sqrt(784))
        # self.bias = nn.Parameter(torch.randn(10))
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        # return xb @ self.weights + self.bias
        return self.lin(xb)

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)


# def nll(inpt, target): #negative log-likelihood
#     return -inpt[range(target.shape[0]), target].mean() #?

# loss_func = nll
loss_func = F.cross_entropy

model, opt = get_model()

bs = 64 #batch size
xb = x_train[:bs]
yb = y_train[:bs]

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1) #((batch,10) -> (batch))
    return (preds == yb).float().mean()

# print(accuracy(preds,yb))

# for epoch in range(epochs):
#     for i in range((n - 1) // bs + 1):
#         start_i = i * bs
#         end_i = start_i + bs
#         xb = x_train[start_i:end_i]
#         yb = y_train[start_i:end_i]
#         pred = model(xb)
#         loss = loss_func(pred, yb)

#         loss.backward()
#         with torch.no_grad():
#             # weights -= weights.grad * lr
#             # bias -= bias.grad * lr
#             # weights.grad.zero_()
#             # bias.grad.zero_()
#             for p in model.parameters():
#                 p -= p.grad * lr
#             model.zero_grad()

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)
# train_dl = DataLoader(train_ds, batch_size=bs)
# valid_dl = DataLoader(valid_ds, batch_size = bs*2)

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs*2)
    )

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)

# def fit():
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):

    for epoch in range(epochs):
        # for i in range((n-1) // bs + 1):
        model.train() # training mode
        for xb, yb in train_dl:
            # # start_i = i*bs
            # # end_i = start_i + bs
            # # xb = x_train[start_i:end_i]
            # # yb = y_train[start_i:end_i]
            # # xb, yb = train_ds[start_i:end_i]
            # pred = model(xb)
            # loss = loss_func(pred, yb)

            # loss.backward()
            # # with torch.no_grad():
            # #     for param in model.parameters():
            # #         param -= param.grad * lr
            # #     model.zero_grad()
            # opt.step()
            # opt.zero_grad()
            loss_batch(model, loss_func, xb, yb, opt)
        model.eval()
        with torch.no_grad():
            # valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
                )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        
        print(epoch, val_loss)

# print(loss_func(model(xb), yb))
# fit()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
# print(loss_func(model(xb), yb), accuracy(model(xb), yb))
# print(loss_func(model(xb), yb))
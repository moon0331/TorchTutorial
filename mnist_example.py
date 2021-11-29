import random

import torch
from torch import nn
from torch import optim
from torch.types import Device
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
from torchvision import transforms

CUDA_AVAILABLE = torch.cuda.is_available()
device = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')

print(f'training in {device}')

SEED = 777

random.seed(SEED)
torch.manual_seed(SEED)
if device == 'cuda':
    torch.cuda.manual_seed_all(SEED)

epochs = 15
batch_size = 100
train_ratio = 0.9

mnist_train = dsets.MNIST('MNIST_data/', 
                                True, 
                                transforms.ToTensor(),
                                download=True)

mnist_test = dsets.MNIST('MNIST_data/', 
                                False, 
                                transforms.ToTensor(),
                                download=True)

data_loader = DataLoader(mnist_train, batch_size, shuffle=True, drop_last=True)

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return torch.Tensor(self.func(x))

class Model(nn.Module): # nn is class, F is function (same as crossentropy)
    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(784, 64)
        self.L3 = nn.Linear(64, 10)

    def forward(self, xb):
        xb = xb.view(-1, 28*28)
        xb = nn.ReLU6()(self.L1(xb))
        xb = self.L3(xb)
        return xb

model = Model()
model.to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
opt = optim.Adagrad(model.parameters(), 0.1)

for epoch in range(epochs):
    model.train()
    avg_cost = 0
    total_batch = len(data_loader)

    for x, y in data_loader:
        # x.to(device)
        # y.to(Device)
        opt.zero_grad()
        y_pred = model(x)
        cost = loss_fn(y_pred, y)
        cost.backward()
        opt.step()
        avg_cost += cost/total_batch

    model.eval()
    # pass
    with torch.no_grad():
        x_test = mnist_test.data.float()
        y_test = mnist_test.targets.float()

        y_test_pred = model(x_test)
        # 레이블 뽑은 뒤 일치하는지 체크
        correct_pred = torch.argmax(y_test_pred, 1) == y_test
        # true == 1
        accuracy = correct_pred.float().mean()
        print(f'test accuracy = {accuracy:4f} ', end='')

        y_train_pred = model(mnist_train.data.float())
        correct_pred = torch.argmax(y_train_pred, 1) == mnist_train.targets.float()
        tacc = correct_pred.float().mean()

    print(f'epoch {epoch}, cost = {avg_cost:4f}, train acc = {tacc:4f}')

'''
mnist.py:69: UserWarning: test_data has been renamed data
  warnings.warn("test_data has been renamed data")
mnist.py:59: UserWarning: test_labels has been renamed targets
  warnings.warn("test_labels has been renamed targets")

-> so let's use 'data' instead of 'test_data' or 'train_data'
   and 'targets' instead of 'train_labels' and 'test_labels'

accuracy = 0.34459999203681946
'''
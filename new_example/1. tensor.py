import numpy as np
import torch
import math

# x = np.linspace(-math.pi, math.pi, 2000)
# y = np.sin(x)

dtype = torch.float
device = torch.device('cpu') #'cuda:0'

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# a, b, c, d = np.random.randn(4) # 4 floats
# a = torch.randn((), device=device, dtype=dtype)
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

lr = 1e-6

for t in range(2000):
    y_pred = a + b*x + c*x**2 + d*x**3

    # loss = np.square(y_pred-y).sum() #(y_pred-y)**2
    # loss = (y_pred - y).pow(2).sum().item() # item() : only one value tensor to scalar
    loss = (y_pred - y).pow(2).sum()

    if t % 100 == 99:
        print(t, loss.item())

    # grad_y_pred = 2.0 * (y_pred - y)
    # grad_a = grad_y_pred.sum()
    # grad_b = (grad_y_pred * x).sum()
    # grad_c = (grad_y_pred * x ** 2).sum()
    # grad_d = (grad_y_pred * x ** 3).sum()
    loss.backward()

    # a -= lr * grad_a
    # b -= lr * grad_b
    # c -= lr * grad_c
    # d -= lr * grad_d

    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad
        c -= lr * c.grad
        d -= lr * d.grad

        a.grad = b.grad = c.grad = d.grad = None



print(f'result: y = {a} + {b}x + {c}x^2 + {d}x^3')
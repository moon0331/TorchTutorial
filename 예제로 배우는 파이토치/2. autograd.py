import torch

dtype = torch.float
device = torch.device('cpu')

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype) # 입력 데이터(랜덤)
y = torch.randn(N, D_out, device=device, dtype=dtype) # 출력 데이터(랜덤)

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

lr = 1e-6

for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    loss.backward() # gradient 계산

    with torch.no_grad():
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()
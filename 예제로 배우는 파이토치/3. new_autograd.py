import torch

class MyRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        return input_.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_ < 0] = 0
        return grad_input

dtype = torch.float
device = torch.device('cpu')

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype) # 입력 데이터(랜덤)
y = torch.randn(N, D_out, device=device, dtype=dtype) # 출력 데이터(랜덤)

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

lr = 1e-6

for t in range(500):
    relu = MyRelu.apply
    y_pred = relu(x.mm(w1)).mm(w2)
    
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    loss.backward() # gradient 계산

    with torch.no_grad():
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()
import torch

print(torch.empty(5, 3))
print(torch.rand(5,3))
x = torch.zeros(5, 33, dtype=torch.long)
print(x.new_ones(5, 3, dtype=torch.double))
print()
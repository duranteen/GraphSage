import torch

a = torch.rand((3, 4))
b = torch.rand((4, 5))

print(b @ a)
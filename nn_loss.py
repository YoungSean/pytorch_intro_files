import torch
from torch import nn

pred = torch.tensor([1., 2, 3])
target = torch.tensor([1., 2, 5])

pred = torch.reshape(pred, [1, 1, 1, 3])
target = torch.reshape(target, [1, 1, 1, 3])

loss = nn.L1Loss()

result = loss(pred, target)
print(result)

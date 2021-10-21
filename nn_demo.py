import torch.nn as nn
import torch


class Sean(nn.Module):

    def __init__(self):
        super(Sean, self).__init__()

    def forward(self, x):
        output = x + 5
        return output


lu = Sean()
x = torch.tensor(2.0)
y = lu(x)
print(y)
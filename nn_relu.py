
import torch
import torch.nn as nn

x = torch.tensor([[1, -0.5],
                  [3, -5.0]],
                 dtype=torch.float32)

class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.relu = nn.ReLU() # invoke the relu function from nn.functional.relu()

    def forward(self, x):
        return self.relu(x)


model = MyModel()
y = model(x)
print(y)
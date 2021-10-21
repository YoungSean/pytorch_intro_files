import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms

x = torch.tensor([[1, 2, 0, 3, 1],
                  [0, 1, 2, 3, 1],
                  [1, 2, 1, 0, 0],
                  [5, 2, 3, 1, 1],
                  [2, 1, 0, 1, 1]], dtype=torch.float32)

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

print(x.shape)
# print(kernel.shape)
x = torch.reshape(x, shape=[1, 1, 5, 5])


# kernel = torch.reshape(kernel, shape=[1, 1, 3, 3])
#
# output = F.conv2d(input,kernel,stride=1,padding=0)
# print(output)

class Sean(nn.Module):

    def __init__(self):
        super(Sean, self).__init__()
        # Do not forget set the ceil mode. Its default value is False.
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), ceil_mode=True)

    def forward(self, x):
        return self.maxpool(x)


lu = Sean()
y = lu(x)
print(y)

test_data = datasets.CIFAR10('D:\\datasets', train=False, transform=transforms.ToTensor())
dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

with SummaryWriter('nnlogs') as writer:
    step = 0
    for data in dataloader:
        imgs, targets = data
        writer.add_images('original', imgs, step)
        out_imgs = lu(imgs)
        writer.add_images('output', out_imgs, step)
        step += 1

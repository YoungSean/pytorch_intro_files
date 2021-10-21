import torch
from torchvision import datasets
import torchvision
from torch import nn

# train_data = datasets.ImageNet('D:\\datasets', split="train",download=True,transform=torchvision.transforms.ToTensor())
train_data = datasets.CIFAR10('D:\\datasets', train=True, download=False, transform=torchvision.transforms.ToTensor())
vgg16_f = torchvision.models.vgg16(pretrained=False)
print(vgg16_f)
vgg16_t = torchvision.models.vgg16(pretrained=True)

vgg16_t.add_module('add_decision', nn.Linear(1000,10))
print(vgg16_t)

print(type(vgg16_f.classifier))

vgg16_f.classifier[6] = nn.Linear(1000,10)
print(vgg16_f)

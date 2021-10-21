import torchvision
import torch


vgg16 = torchvision.models.vgg16(pretrained=False)

# sava model 1
torch.save(vgg16, "myvgg16_first.pth")
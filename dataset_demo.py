import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

train_data = torchvision.datasets.CIFAR10(root='D:\\datasets', train=True, download=False)
test_data = torchvision.datasets.CIFAR10(root='D:\\datasets', train=False, transform=ToTensor(), download=False)

# print(test_data[0])
# img, target = test_data[0]
#
# print(target)
# img.show()
print(len(test_data))
test_loader = DataLoader(test_data, batch_size=64, shuffle=True,  num_workers=0, drop_last=True)
writer = SummaryWriter("logs")
step = 0
for epoch in range(2):
    for data in test_loader:
        imgs, targets = data
        writer.add_images("test_data_{}".format(epoch), imgs, step)
        step += 1


writer.close()
from torch.utils.tensorboard import SummaryWriter
import cv2

from torchvision import transforms

img_path = "images/shanyi.jpg"
img_arr = cv2.imread(img_path)

trans_totensor = transforms.ToTensor()
img_ts = trans_totensor(img_arr)
# Normalize: (pixel - mean) / std
normalize = transforms.Normalize([0.5,0.5,0.5],[3,4,5])
img_norm = normalize(img_ts)

# resize
# input: PIL image
print(img_ts.shape)
resize = transforms.Resize([300,450])
img_small = resize(img_ts)
print(img_small.shape)

# random crop
trans_crop = transforms.RandomCrop(300)
trans_comps = transforms.Compose([trans_crop])

img_crop = trans_comps(img_ts)

with SummaryWriter('logs') as writer:
    writer.add_image('small', img_crop,2)
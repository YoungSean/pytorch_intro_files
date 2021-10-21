from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

img_path = "hymenoptera_data/train/bees/85112639_6e860b0469.jpg"
img_pil = Image.open(img_path)

# 实例化了一个class为具体的工具，为什么用class而不是function来定义工具呢？
# 我猜是 为了定制工具，可以加入各种自己的调整
tensor_trans = transforms.ToTensor()
# 使用工具来转化image PIL 为tensor形式
img_ts = tensor_trans(img_pil)
print(img_ts)

with SummaryWriter('transLogs') as writer:
    writer.add_image("Tensor Image", img_ts)
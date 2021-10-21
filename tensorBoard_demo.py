from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
img_path = "hymenoptera_data/train/bees/21399619_3e61e5bb6f.jpg"
img = Image.open(img_path)
img_arr = np.array(img)


with SummaryWriter("logs") as writer:
    # for i in range(100):
    #     writer.add_scalar("y=2x", 2*i,i)
    writer.add_image("show an image", img_arr, 1, dataformats='HWC')



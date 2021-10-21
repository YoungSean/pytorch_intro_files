from torch.utils.data import Dataset
import os
from PIL import Image

class Mydata(Dataset):

    def __init__(self, root_dir, label):
        self.root = root_dir
        self.label = label
        self.path = os.path.join(self.root, self.label)
        self.fnames = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.fnames[idx]
        img_path = os.path.join(self.path, img_name)
        label = self.label
        img = Image.open(img_path)
        return img, label

    def __len__(self):
        return len(self.fnames)

root_dir = "hymenoptera_data/train"
ants_label = "ants"
bees_label = "bees"
ant_data = Mydata(root_dir, ants_label)
img, label = ant_data[1]
img.show()

bee_data = Mydata(root_dir, bees_label)
all_data = ant_data + bee_data
print(len(all_data))
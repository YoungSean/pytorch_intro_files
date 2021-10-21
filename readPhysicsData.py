import torch
from torch.utils.data import Dataset
import time
import os
from PIL import Image
import numpy as np
import healpy as hp
from pathlib import Path
from torchvision.transforms import ToTensor
import torchvision
from torch.utils.data import DataLoader
import datetime

###############
SIM_DIR = Path(
    "/data/lab/YangxiaoLu/physicsImages/nops8/outputnops512_updateddust")


def convert(seconds):
    '''Transform seconds to hours or minutes'''
    days = seconds // (24 * 3600)
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d:%02d" % (days, hour, minutes, seconds)


start_time_str = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
start_time = time.time()
print(f"start from {start_time} \n")
# os.makedirs(f"logs_lyx/{start_time_str}")


##############
SIM_DIR = Path(
    "/data/lab/YangxiaoLu/physicsImages/nops8/outputnops512_updateddust")
base_dir = Path('.')
file_path = Path('./sim0001')
planck_map_freqs = [70, 100, 143, 217, 353, 545, 857]


def get_raw_sample(first_example):
    i = 0
    y_raw = hp.read_map(
        SIM_DIR / f"sim{first_example + i:04}/cmb.fits",
        # base_dir / f"sim{i:04}/cmb.fits",
        verbose=False,
        dtype=np.float32,
        nest=True,
    )
    x_raw = []
    for j, freq in enumerate([f"{freq:03d}" for freq in planck_map_freqs]):
        x_raw.append(
            hp.read_map(
                SIM_DIR / f"sim{first_example + i:04}/{freq}.fits",
                # base_dir / f"sim{i:04}/{freq}.fits",
                verbose=False,
                dtype=np.float32,
                nest=True,
            )
        )
    x_raw = np.array(x_raw)
    return (x_raw, y_raw)


# # x_raw, y_raw = get_raw_sample(1)
# print(x_raw.shape)
# print(y_raw.shape)


crop_size = 28 * 28


def crop_sample(start, num_chunk, image, label, size):
    result = []
    for i in range(num_chunk):
        x_chunk = image[:, start:start + size].reshape((-1, 28, 28))
        y_chunk = label[start:start + size].reshape((-1, 28, 28))
        if i == 0:
            print("x_chunk size is {}".format(x_chunk.shape))
            print("y_chunk size is {}".format(y_chunk.shape))
        result.append((x_chunk, y_chunk))
        start = start + size
    return result


# print(samples[0][0])
# print(samples[1][0])
def get_multiple_samples(num):
    samples = []
    for i in range(num):
        x_raw, y_raw = get_raw_sample(i)
        samples += crop_sample(1000, 20, x_raw, y_raw, size=crop_size)
    return samples


samples = get_multiple_samples(200)
min_x = float('inf')
max_x = -float('inf')
for i in samples:
    x_arr = np.array(i[0])
    min_x = min(min_x, np.min(x_arr))
    max_x = max(max_x, np.max(x_arr))

print("The minimum number of input chunks in the samples: ", min_x)
print("The maximum number of input chunks in the samples: ", max_x)

class Mydata(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = torch.from_numpy(sample[0])
        label = torch.from_numpy(sample[1])
        return img, label

    def __len__(self):
        return len(self.samples)


physics_data = Mydata(samples)
img, label = physics_data[0]
# print(img)

training_loader = DataLoader(physics_data, batch_size=20, shuffle=True)

### Model Part
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')
from model_vae2 import VariationalAutoencoder

vae = VariationalAutoencoder(latent_dims=256, device=device)
lr = 1e-3

optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)

vae.to(device)


#########Training
def train_epoch(vae, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for x, _ in dataloader:
        # Move tensor to the proper device
        x = x.to(device)
        x_hat = vae(x)
        # Evaluate loss
        loss = ((x - x_hat) ** 2).sum() + vae.encoder.kl

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.item()))
        train_loss += loss.item()

    return train_loss / len(dataloader.dataset)


num_epochs = 1000

for epoch in range(num_epochs):
    train_loss = train_epoch(vae, device, training_loader, optim)
    # val_loss = test_epoch(vae,device,valid_loader)
    # print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))
    if epoch % 50 == 0:
        print('\n EPOCH {}/{} \t train loss {:.3f}'.format(epoch + 1, num_epochs, train_loss))

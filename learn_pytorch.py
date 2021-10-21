import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

np_data = np.arange(12).reshape((3, 4))
t_data = torch.from_numpy(np_data)

tensor2arr = t_data.numpy()

print("Numpy data: \n", np_data, "\nTorch data: \n", t_data)
print("back to array: \n", tensor2arr)
print(tensor2arr.dtype)
torch.cuda.is_available()
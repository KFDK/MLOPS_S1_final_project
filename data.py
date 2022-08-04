import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import os
import torchvision.transforms as transforms
import pdb
from pathlib import Path
import glob

"""
def mnist():
    # exchange with the corrupted mnist dataset
    train = torch.randn(50000, 784)
    test = torch.randn(10000, 784) 
    return train, test
"""


class MNIST_corrupted(Dataset):
    def __init__(self, data_path, data_type, transform=None, seed=1234):
        # data loading
        self.data_path = data_path
        # self.files = sorted(glob.glob(data_path + "/*.npz"))
        # pdb.set_trace()
        self.transform = transform
        self.data_type = data_type
        self.seed = seed

    def __getitem__(self, index):
        set = np.random.randint(low=0, high=4)  # randomly choose from datasets
        # indexing in dataset
        if self.data_type == "train":
            img_path = os.path.join(self.data_path, "train_" + str(set) + ".npz")

        if self.data_type == "test":
            img_path = os.path.join(self.data_path, "test.npz")

        image = np.load(img_path)["images"][index]

        label = torch.tensor(np.load(img_path)["labels"][index])

        if self.transform:
            image = self.transform(image)
            image = image.type(torch.FloatTensor)

        return image, label

    def __len__(self):
        # find length of dataset
        return 5000


"""
data_path = "./data/corruptmnist/"
transform = transforms.Compose([transforms.ToTensor()])
data_type = "train"
dataset = MNIST_corrupted(data_path=data_path, data_type=data_type, transform=transform)
"""

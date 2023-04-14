import glob
import os
import torch
import torchvision
import numpy as np

from torch.utils.data import Dataset
from .parameters import image_size, imagenet_mean_std


def get_stl10(root, split, subset_size=None):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.Normalize(*imagenet_mean_std),
    ])
    dataset = torchvision.datasets.STL10(root=root, split=split, transform=transform, download=True)
    if subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_size)))
    return dataset


def get_2022_lens_geoff(root, subset_size=None):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # grayscale -> RGB
        torchvision.transforms.Normalize(*imagenet_mean_std),
    ])
    dataset = DeepLenstronomyDataset(root=root, transform=transform)
    if subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_size)))
    return dataset


class DeepLenstronomyDataset(Dataset):
    """ The DeepLenstronomyDataset class.
    Args:
        Dataset: torch.utils.data.Dataset class
    """
    def __init__(self, root, transform=None):
        """ Initialize the class.

        Args:
            root (str): dir of the dataset
            transform (torchvision.transforms, optional): transforms for images. Defaults to None.
        """
        self.root = root
        self.transform = transform
        self.file_names = glob.glob(os.path.join(self.root, "*.npy"))
        self.size = len(self.file_names)

    def __getitem__(self, index):
        if index >= self.size:
            raise Exception
        image = np.load(self.file_names[index]).astype("float32")
        image = self.transform(image)
        fake_label = torch.zeros(image.shape[0], 1)  # None because we don't care about labels
        return image, fake_label
        
    def __len__(self):
        return self.size

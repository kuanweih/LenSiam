import glob
import os
import torch
import torchvision
import numpy as np

from PIL import Image
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
    transform = CommonLensingTransform()
    dataset = DeepLenstronomyDataset(root=root, transform=transform)
    if subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_size)))
    return dataset


def get_real_hst(root, subset_size=None, suffix=None):
    transform = CommonLensingTransform()
    dataset = RealHSTDataset(root=root, transform=transform, suffix=suffix)
    if subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_size)))
    return dataset


def get_galaxy_zoo(root, subset_size=None):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(
            lambda x: x / 255.),  # normalize pixel values from [0, 255] to [0, 1]
        torchvision.transforms.Normalize(*imagenet_mean_std),
    ])
    dataset = GalaxyZooDataset(root=root, transform=transform)
    if subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_size)))
    return dataset


class CommonLensingTransform():
    """ Pytorch transform class for a single image.
    """
    def __init__(self):
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.Lambda(
                lambda x:(x - x.min()) / (x.max() - x.min())),  # normalize pixel values in [0, 1]
            torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # grayscale -> RGB
            torchvision.transforms.Normalize(*imagenet_mean_std),
        ])
    def __call__(self, x):
        return self.transform(x)


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


class RealHSTDataset(Dataset):
    """ The RealHSTDataset class for data downloaded and processed by
        https://github.com/kuanweih/lensed_quasar_database_scraper

    Args:
        Dataset: torch.utils.data.Dataset class
    """
    def __init__(self, root, transform=None, suffix=None):
        """ Initialize the class.

        Args:
            root (str): dir of the dataset
            transform (torchvision.transforms, optional): transforms for images. Defaults to None.
            suffix (str, optional): suffix of the file names to be loaded. Defaults to None.
                                    Usually suffix = 'cutout' will be used for good cutout images.
        """
        self.root = root
        self.transform = transform
        pattern = "*.npy" if suffix is None else f"*{suffix}.npy"
        self.file_names = glob.glob(os.path.join(self.root, pattern))
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


class GalaxyZooDataset(Dataset):
    """ The GalaxyZooDataset class for data downloaded from
        https://www.kaggle.com/competitions/galaxy-zoo-the-galaxy-challenge/overview

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
        self.file_names = glob.glob(os.path.join(self.root, "*.jpg"))
        self.size = len(self.file_names)

    def __getitem__(self, index):
        if index >= self.size:
            raise Exception
        image = Image.open(self.file_names[index])
        image = image.resize((image_size, image_size))
        image = np.array(image).astype("float32")
        image = self.transform(image)
        fake_label = torch.zeros(image.shape[0], 1)  # None because we don't care about labels
        return image, fake_label

    def __len__(self):
        return self.size

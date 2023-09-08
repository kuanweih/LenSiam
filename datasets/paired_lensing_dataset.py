import glob
import os
import numpy as np
import random
from PIL import Image

from astropy.io import fits
from torch.utils.data import Dataset
from torchvision import transforms

from .parameters import image_size, imagenet_mean_std


target_keys = ['theta_E', 'e1', 'e2', 'center_x', 'center_y', 'gamma', 'gamma1', 'gamma2']


def load_fits_file(file_path):
    """ Helper function to load fits file
    Args:
        file_path (str): 
    Returns:
        data (numpy.ndarray): paired lensing images as an array of shape (110, 110, 2)
        param (dict): dictionary of lensing parameters/labels
    """
    with fits.open(file_path, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float32)
        param = {}
        for key in target_keys:
            param[key] = hdul[0].header[key]
    return data, param


class LensingImageTransform():
    """ Pytorch transform class for a single image.
    """
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # grayscale -> RGB
            transforms.Normalize(*imagenet_mean_std),
        ])
    def __call__(self, x):
        return self.transform(x)


class SimSiamTransform():
    def __init__(self):
        # image_size = 224 if image_size is None else image_size # by default simsiam use image size 224
        p_blur = 0.5 if image_size > 32 else 0 # exclude cifar
        # the paper didn't specify this, feel free to change this value
        # I use the setting from simclr which is 50% chance applying the gaussian blur
        # the 32 is prepared for cifar training where they disabled gaussian blur
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # grayscale -> RGB
            transforms.Normalize(*imagenet_mean_std)
        ])
    def __call__(self, x):
        x = Image.fromarray((x * 255).astype(np.uint8))  # Convert NumPy array to PIL Image for the input of random augmentation transformation.
        return self.transform(x)


class PairedLensingImageDataset(Dataset):
    """ Pytorch Dataset Object for the paired lensing image dataset in fits file.
    """
    def __init__(self, root=None, aug_method=None):
        self.root = root
        self.aug_method = aug_method
        self.file_names = glob.glob(os.path.join(self.root, "*.fits"))
        self.size = len(self.file_names)

    def __getitem__(self, idx):
        if idx >= self.size:
            raise Exception
        file_path = self.file_names[idx]
        img_pair, label = load_fits_file(file_path)

        if self.aug_method == 'lensiam':
            transform = LensingImageTransform()
            img1 = transform(img_pair[:, :, 0])
            img2 = transform(img_pair[:, :, 1])
            return img1, img2, label, file_path
        elif self.aug_method == 'simsiam':
            transform_aug = SimSiamTransform()
            # Randomly choose one of the images from img_pair
            i = random.choice([0, 1])
            img1 = transform_aug(img_pair[:, :, i])
            img2 = transform_aug(img_pair[:, :, i])
            return img1, img2, label, file_path
        else:
            raise NotImplementedError
        
    
    def __len__(self):
        return self.size

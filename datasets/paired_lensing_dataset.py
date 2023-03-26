import glob
import os
import numpy as np
from astropy.io import fits
from torch.utils.data import Dataset
from torchvision import transforms


parameters = ['theta_E', 'e1', 'e2', 'center_x', 'center_y', 'gamma', 'gamma1', 'gamma2']

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
        for key in parameters:
            param[key] = hdul[0].header[key]
    return data, param


class LensingImageTransform():
    """ Pytorch transform class for a single image.
    """
    def __init__(self):
        image_size = 224
        imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # grayscale -> RGB
            transforms.Normalize(*imagenet_mean_std),
        ])
    def __call__(self, x):
        return self.transform(x)


class PairedLensingImageDataset(Dataset):
    """ Pytorch Dataset Object for the paired lensing image dataset in fits file.
    """
    def __init__(self, root=None):
        self.root = root
        self.file_names = glob.glob(os.path.join(self.root, "*.fits"))
        self.size = len(self.file_names)

    def __getitem__(self, idx):
        if idx >= self.size:
            raise Exception
        img_pair, label = load_fits_file(self.file_names[idx])
        transform = LensingImageTransform()
        img1 = transform(img_pair[:, :, 0])
        img2 = transform(img_pair[:, :, 1])
        return img1, img2, label

    def __len__(self):
        return self.size

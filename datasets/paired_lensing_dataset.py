import glob
import os
from astropy.io import fits
from torch.utils.data import Dataset


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
        data = hdul[0].data
        param = {}
        for key in parameters:
            param[key] = hdul[0].header[key]
    return data, param


class PairedLensingDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.file_names = glob.glob(os.path.join(self.root, "*.fits"))
        self.size = len(self.file_names)

    def __getitem__(self, idx):
        if idx >= self.size:
            raise Exception

        img_pair, label = load_fits_file(self.file_names[idx])

        return img_pair[:,:,0], img_pair[:,:,1], label
        # return [torch.randn((3, 224, 224)), torch.randn((3, 224, 224))], [0,0,0]

    def __len__(self):
        return self.size

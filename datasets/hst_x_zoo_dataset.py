import glob
import os
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from .parameters import image_size, imagenet_mean_std


def get_hst_x_zoo(root, subset_size=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.Lambda(lambda x:(x - x.min()) / (x.max() - x.min())),  # normalized [0, 1]
        transforms.Normalize(*imagenet_mean_std),
    ])
    dataset = HSTxZOODataset(root=root, transform=transform)
    if subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_size)))
    return dataset



class HSTxZOODataset(Dataset):
    """ The joint set of HST dataset and Galaxy Zoo dataset created by
        https://github.com/kuanweih/lensed_quasar_database_scraper

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
        self.class_map = {"zoo" : 0, "hst": 1}
        self.data = []
        self.class_size = {"zoo": 0, "hst": 0}  # Keep track of the number of images per class
        for class_path in glob.glob(os.path.join(self.root, "*")):
            if not os.path.isdir(class_path):
                continue
            class_name = os.path.basename(class_path)
            for img_path in glob.glob(os.path.join(class_path, "*.jpg")):
                self.data.append([img_path, class_name])
                self.class_size[class_name] += 1  # Increase the count of the class
        self.size = len(self.data)

    def __getitem__(self, index):
        if index >= self.size:
            raise Exception
        img_path, class_name = self.data[index]
        image = Image.open(img_path)
        image = image.resize((image_size, image_size))
        image = np.array(image).astype("float32")
        image = self.transform(image)
        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id)
        class_id = torch.nn.functional.one_hot(class_id, num_classes = 2)
        # class_id = torch.tensor(class_id, dtype=torch.long)
        return image, class_id

    def __len__(self):
        return self.size

    def get_class_size(self, class_name):
        return self.class_size.get(class_name, 0)
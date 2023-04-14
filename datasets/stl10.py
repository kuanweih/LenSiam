import torch
import torchvision
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

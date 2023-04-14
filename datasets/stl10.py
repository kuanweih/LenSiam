import torch
import torchvision


def get_stl10(root, split, subset_size=None):
    image_size = 224
    imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.Normalize(*imagenet_mean_std),
    ])
    dataset = torchvision.datasets.STL10(root=root, split=split, transform=transform, download=True)
    if subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_size)))
    return dataset


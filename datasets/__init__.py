import torch

from .paired_lensing_dataset import PairedLensingImageDataset
from .umap_testsets import get_stl10, get_2022_lens_geoff, get_real_hst


def get_dataset(dataset_name, data_dir, subset_size=None, aug_method=None):
    if dataset_name == 'paired-lensing':
        dataset = PairedLensingImageDataset(data_dir, aug_method=aug_method)
    else:
        raise NotImplementedError
    if subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_size)))
    return dataset


def get_umap_testset(dataset_name, **kwarg):
    if dataset_name == 'STL10':
        dataset = get_stl10(**kwarg)
    elif dataset_name == 'Lens2022':
        dataset = get_2022_lens_geoff(**kwarg)
    elif dataset_name == 'RealHST':
        dataset = get_real_hst(**kwarg)
    else:
        raise NotImplementedError
    return dataset

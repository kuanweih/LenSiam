from .paired_lensing_dataset import PairedLensingImageDataset
from .stl10 import get_stl10


def get_dataset(dataset_name, data_dir):
    if dataset_name == 'paired-lensing':
        dataset = PairedLensingImageDataset(data_dir)
    else:
        raise NotImplementedError
    return dataset


def get_umap_testset(dataset_name, **kwarg):
    if dataset_name == 'STL10':
        dataset = get_stl10(**kwarg)
    else:
        raise NotImplementedError
    return dataset

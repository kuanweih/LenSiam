from .paired_lensing_dataset import PairedLensingDataset


def get_dataset(dataset, data_dir, transform, debug_subset_size=None):
    if dataset == 'lensing-dev':
        dataset = PairedLensingDataset(data_dir, transform=transform)
    else:
        raise NotImplementedError
    return dataset
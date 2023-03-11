from .paired_lensing_dataset import PairedLensingImageDataset


def get_dataset(dataset, data_dir):
    if dataset == 'lensing-dev':
        dataset = PairedLensingImageDataset(data_dir)
    else:
        raise NotImplementedError
    return dataset

from .paired_lensing_dataset import PairedLensingImageDataset


def get_dataset(dataset_name, data_dir):
    if dataset_name == 'lensing-dev':
        dataset = PairedLensingImageDataset(data_dir)
    else:
        raise NotImplementedError
    return dataset

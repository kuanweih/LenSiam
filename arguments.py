import argparse
import os
import torch
import numpy as np
import random
import re 
import yaml
import shutil

from datetime import datetime


class Namespace:
    def __init__(self, somedict):
        for key, value in somedict.items():
            # Assert: (1) type(key) is str, and (2) any single character in key is 
            # either a letter (uppercase or lowercase), an underscore, or a hyphen.
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value
    
    def __getattr__(self, attribute):
        raise AttributeError(f"Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!")


def set_deterministic(seed):
    """ Set all the random seeds to seed to make it deterministic.
    Args:
        seed (int): random seed
    """
    # seed by default is None 
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 

def get_args():
    """ Get arguments.
    Returns:
        argparse.Namespace: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', required=True, type=str, help="xxx.yaml")
    parser.add_argument('--data_dir', required=True, type=str, help="path/to/the/dataset")
    parser.add_argument('--log_dir', type=str, default="./outputs/logs")
    parser.add_argument('--ckpt_dir', type=str, default="./outputs/ckpts")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # load config yaml to args
    with open(args.config_file, 'r') as f:
        for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
            vars(args)[key] = value

    # make sure the following arg values are not None
    assert not None in [args.log_dir, args.data_dir, args.ckpt_dir, args.name]

    # create folders/files: log, checkpoint, config, etc
    args.log_dir = os.path.join(
        args.log_dir, f"in-progress_{datetime.now().strftime('%m%d%H%M%S')}_{args.name}")
    os.makedirs(args.log_dir, exist_ok=False)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    shutil.copy2(args.config_file, args.log_dir)
    
    # set most if not all random seeds by args.seed if it is provided
    set_deterministic(args.seed)

    # create kwargs for various purposes
    vars(args)['dataset_kwargs'] = {
        'dataset_name': args.dataset.name,
        'data_dir': args.data_dir,
    }
    vars(args)['dataloader_kwargs'] = {
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.dataset.num_workers,
    }

    return args

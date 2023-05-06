import os
import torch
import umap
import numpy as np

import torch.nn as nn

import yaml
import shutil
import argparse


from collections import defaultdict
from tqdm import tqdm
from arguments import get_args_umap
from datasets import get_dataset, get_umap_testset
from models import get_backbone


# # TODO mv
# # Binary Classification Model
# class BinaryClassifier(nn.Module):

#     def __init__(self, input_size, output_size):

#         # output_size = 2. 0: not lens. 1: lens.

#         super(BinaryClassifier, self).__init__()
#         self.fc = nn.Linear(input_size, output_size)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.fc(x)
#         x = self.sigmoid(x)
#         return x


def main(device, config):

    # # Load dataset
    # dataset = get_dataset(args.dataset.name, args.dataset.data_dir, args.dataset.subset_size)
    # data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.train.batch_size)

    # Load the trained backbone model
    model = get_backbone(config["model"]["backbone"]).to(device)
    ckpt = torch.load(config["model"]["file"], map_location='cpu')
    assert ckpt["backbone"] == config["model"]["backbone"]  # make sure loaded model == model
    model.load_state_dict(ckpt["backbone_state_dict"])


    print(model.encoder.ln)

    model.heads = nn.Sequential(
        nn.Linear(768, 2),  # output_size = 2. 0: not lens. 1: lens.
        nn.Sigmoid(),
    )

    print(model.heads)

    
    # model = torch.nn.DataParallel(model)





#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)


















#     # Forward pass to get the learned representation
#     dict_result = defaultdict(list)
#     with torch.no_grad():
#         for idx, (images1, images2, labels) in enumerate(tqdm(data_loader, desc="Train Set")):
#             # Get representations
#             repr1 = model.forward(images1.to(device, non_blocking=True))
#             repr2 = model.forward(images2.to(device, non_blocking=True))
#             dict_result["representation"].extend(torch.concat([repr1, repr2]).cpu().tolist())
#             # Get labels
#             for key, val in labels.items():
#                 val = val.cpu().tolist()
#                 dict_result[key].extend(val)
#                 dict_result[key].extend(val)

#     # Convert list to np arrays
#     for key, val in dict_result.items():
#         dict_result[key] = np.array(val)

#     # Use the fitted reducer to calculate UMAP embeddings for the UMAP testsets
#     dict_testset_repr = {}
#     for key in vars(args.testsets):
#         _kwarg = vars(vars(args.testsets)[key])
#         dataset = get_umap_testset(key, **_kwarg)
#         dict_testset_repr[key] = calc_representations_testset(dataset, model, args, device, key)

#     # Fit the UMAP reducer using all data points (main + testsets)
#     reducer = umap.UMAP(n_neighbors=args.umap.n_neighbors)
#     data = np.concatenate(
#         [dict_result["representation"]] + [dict_testset_repr[key] for key in dict_testset_repr])
#     reducer.fit(data)
#     del data

#     # Calculate the UMAP embeddings for main dataset
#     dict_result["embeddings"] = reducer.transform(dict_result["representation"])
#     del dict_result["representation"]
#     np.save(os.path.join(args.output_dir, "umap_result.npy"), dict_result)

#     # Calculate the UMAP embeddings for all testsets
#     result = {}
#     for key, representation in dict_testset_repr.items():
#         result[key] = reducer.transform(representation)
#     del dict_testset_repr
#     np.save(os.path.join(args.output_dir, "umap_testsets.npy"), result)


# def calc_representations_testset(dataset, model, args, device, name):
#     """ Calculate representations for a given testset via forward pass to the model

#     Args:
#         dataset (torch.utils.data.Dataset): the test dataset for UMAP
#         model (torch model): the trained model
#         args (argparse.Namespace): args
#         device (str): args.device
#         name (str): dataset name

#     Returns:
#         numpy.ndarray: the representations for the given testset
#     """
#     representations = []
#     data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.train.batch_size)
#     with torch.no_grad():
#         for images, labels in tqdm(data_loader, desc=name):
#             repr = model.forward(images.to(device, non_blocking=True))
#             representations.extend(repr.cpu().tolist())
#     representations = np.array(representations)
#     return representations




if __name__ == "__main__":

    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', required=True, type=str, help="xxx.yaml")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load config
    with open(args.config_file) as file:
        config = yaml.safe_load(file)

    print(f"We will be using device = {args.device}!")

    main(device=args.device, config=config)

    # # Wrap up logs
    # completed_output_dir = args.output_dir.replace('in-progress', 'completed')
    # os.rename(args.output_dir, completed_output_dir)
    # print(f'Output has been saved to {completed_output_dir}')

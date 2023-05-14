import os
import torch
import umap
import numpy as np

from collections import defaultdict
from tqdm import tqdm
from arguments import get_args_umap
from datasets import get_dataset, get_umap_testset
from models import get_backbone


def main(device, args):

    # Load dataset
    dataset = get_dataset(args.dataset.name, args.dataset.data_dir, args.dataset.subset_size)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.train.batch_size)

    # Load the trained model
    flag_hugginface_model = False  # hugginface model calling returns differently
    # - for simsiam backbones trained from this repo
    if args.model.name == "simsiam":
        model = get_backbone(args.model.backbone).to(device)
        ckpt = torch.load(args.model.file, map_location='cpu')
        assert ckpt["backbone"] == args.model.backbone  # make sure loaded model == model
        model.load_state_dict(ckpt["backbone_state_dict"])
    # - for models trained from https://github.com/kuanweih/strong_lensing_vit_resnet
    elif args.model.name == "eccv_2022":
        import torchvision
        from transformers import ViTForImageClassification
        model = torch.load(args.model.file, map_location='cpu')
        if isinstance(model, ViTForImageClassification):
            model.classifier = torch.nn.Identity()
            flag_hugginface_model = True
        elif isinstance(model, torchvision.models.resnet.ResNet):
            model.fc = torch.nn.Identity()
        else:
            raise TypeError
    else:
        raise NotImplementedError

    model = torch.nn.DataParallel(model)
    model.eval()

    # Forward pass to get the learned representation
    dict_result = defaultdict(list)
    with torch.no_grad():
        for (images1, images2, labels, paths) in tqdm(data_loader, desc="Train Set"):
            # Get representations
            repr1 = model(images1.to(device, non_blocking=True))
            repr2 = model(images2.to(device, non_blocking=True))
            if flag_hugginface_model:
                repr1 = repr1[0]
                repr2 = repr2[0]
            dict_result["representation"].extend(torch.concat([repr1, repr2]).cpu().tolist())
            dict_result["path"].extend(list(paths))
            # Get labels
            for key, val in labels.items():
                val = val.cpu().tolist()
                dict_result[key].extend(val)
                dict_result[key].extend(val)

    # Convert list to np arrays
    for key, val in dict_result.items():
        dict_result[key] = np.array(val)

    # Use the fitted reducer to calculate UMAP embeddings for the UMAP testsets
    dict_testset_repr = {}
    for key in vars(args.testsets):
        _kwarg = vars(vars(args.testsets)[key])
        dataset = get_umap_testset(key, **_kwarg)
        dict_testset_repr[key] = calc_representations_testset(
            dataset, model, args, device, key, flag_hugginface_model)

    # Fit the UMAP reducer using all data points (main + testsets)
    reducer = umap.UMAP(n_neighbors=args.umap.n_neighbors)
    data = np.concatenate(
        [dict_result["representation"]] + [dict_testset_repr[key] for key in dict_testset_repr])
    reducer.fit(data)
    del data

    # Calculate the UMAP embeddings for main dataset
    dict_result["embeddings"] = reducer.transform(dict_result["representation"])
    del dict_result["representation"]
    np.save(os.path.join(args.output_dir, "umap_result.npy"), dict_result)

    # Calculate the UMAP embeddings for all testsets
    result = {}
    for key, representation in dict_testset_repr.items():
        result[key] = reducer.transform(representation)
    del dict_testset_repr
    np.save(os.path.join(args.output_dir, "umap_testsets.npy"), result)


def calc_representations_testset(dataset, model, args, device, name, flag_hugginface_model):
    """ Calculate representations for a given testset via forward pass to the model

    Args:
        dataset (torch.utils.data.Dataset): the test dataset for UMAP
        model (torch model): the trained model
        args (argparse.Namespace): args
        device (str): args.device
        name (str): dataset name

    Returns:
        numpy.ndarray: the representations for the given testset
    """
    representations = []
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.train.batch_size)
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=name):
            repr = model(images.to(device, non_blocking=True))
            if flag_hugginface_model:
                repr = repr[0]
            representations.extend(repr.cpu().tolist())
    representations = np.array(representations)
    return representations




if __name__ == "__main__":

    args = get_args_umap()
    print(f"We will be using device = {args.device}!")

    main(device=args.device, args=args)

    # Wrap up logs
    completed_output_dir = args.output_dir.replace('in-progress', 'completed')
    os.rename(args.output_dir, completed_output_dir)
    print(f'Output has been saved to {completed_output_dir}')

import os
import torch
import umap
import numpy as np

from collections import defaultdict
from tqdm import tqdm
from arguments import get_args_umap
from datasets import get_dataset
from models import get_backbone


def main(device, args):

    # Load dataset
    dataset = get_dataset(args.dataset.name, args.dataset.data_dir)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.train.batch_size)

    # Load the trained backbone model
    model = get_backbone(args.model.backbone).to(device)
    model = torch.nn.DataParallel(model)
    ckpt = torch.load(args.model.file, map_location='cpu')
    assert ckpt["backbone"] == args.model.backbone  # make sure loaded model == model
    model.module.load_state_dict(ckpt["backbone_state_dict"])
    model.eval()

    # Forward pass to get the learned representation
    dict_result = defaultdict(list)
    with torch.no_grad():
        for idx, (images1, images2, labels) in enumerate(tqdm(data_loader)):
            # Get representations
            repr1 = model.forward(images1.to(device, non_blocking=True))
            repr2 = model.forward(images2.to(device, non_blocking=True))
            dict_result["representation"].extend(torch.concat([repr1, repr2]).cpu().tolist())
            # Get labels
            for key, val in labels.items():
                val = val.cpu().tolist()
                dict_result[key].extend(val)
                dict_result[key].extend(val)

    # Convert list to np arrays
    for key, val in dict_result.items():
        dict_result[key] = np.array(val)

    # Calculate the UMAP embeddings and save it
    dict_result["embeddings"] = umap.UMAP().fit_transform(dict_result["representation"])
    del dict_result["representation"]
    file_path = os.path.join(args.output_dir, "umap_result.npy")
    np.save(file_path, dict_result)




if __name__ == "__main__":

    args = get_args_umap()
    print(f"We will be using device = {args.device}!")

    main(device=args.device, args=args)

    # Wrap up logs
    completed_output_dir = args.output_dir.replace('in-progress', 'completed')
    os.rename(args.output_dir, completed_output_dir)
    print(f'Output has been saved to {completed_output_dir}')















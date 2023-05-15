import os
import torch
import numpy as np

import torch.nn as nn

import yaml
import shutil
import argparse

from tqdm import tqdm
from models import get_backbone

from datasets.hst_x_zoo_dataset import get_hst_x_zoo



def main(device, config):

    # Load dataset
    dataset = get_hst_x_zoo(
        config["dataset"]["data_dir"],
        subset_size=config["dataset"]["subset_size"],
    )

    train_size = int(config["train"]["train_fraction"] * len(dataset))
    val_size = int(config["train"]["val_fraction"] * len(dataset))
    test_size = len(dataset) - (train_size + val_size)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])

    # Create data loaders for train, val, and test sets
    train_data_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        shuffle = True,
        batch_size = config["train"]["batch_size"],
    )

    val_data_loader = torch.utils.data.DataLoader(
        dataset = val_dataset,
        shuffle = False,
        batch_size = config["train"]["batch_size"],    ## What is the batch size here?
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        shuffle = False,
        batch_size = config["train"]["batch_size"],    ## What is the batch size here?
    )


    # Load the trained backbone model
    model = get_backbone(config["model"]["backbone"]).to(device)
    ckpt = torch.load(config["model"]["file"], map_location='cpu')
    assert ckpt["backbone"] == config["model"]["backbone"]  # make sure loaded model == model
    model.load_state_dict(ckpt["backbone_state_dict"])


    # # Freeze parameters of the pre-trained model  
    # for param in model.parameters():
    #     param.requires_grad = False


    # Dimension for the last layer
    input_dim = model.encoder.ln.normalized_shape[0]  # This will be 768 from our setting
    output_dim = 2  # Binary classification (0: not lens. 1: lens)

    # Change the last layer of the original model
    model.heads = nn.Sequential(
        nn.Linear(input_dim, output_dim),  
        # nn.Sigmoid(),  # Use BCEWithLogitsLoss so no need for sigmoid here
    )    
    model = model.to(device)
    model = torch.nn.DataParallel(model)


    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])

    # Define the loss for the binary classification with imbalanced data
    false_class_size = dataset.get_class_size('zoo')  # 'false_class_size' and 'true_class_size' should add up to len(dataset) 
    true_class_size = dataset.get_class_size('hst')

    # pos_weight = 1. / torch.Tensor([false_class_size, true_class_size]) 
    # pos_weight = pos_weight / pos_weight.sum()

    class_counts = torch.tensor([false_class_size, true_class_size])
    pos_weight = (len(dataset) - class_counts) / class_counts 

    criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)



    global_progress = tqdm(range(0, config["train"]["num_epochs"]), desc='Training')

    for epoch in global_progress:
        model.train()

        local_progress = tqdm(
            train_data_loader,
            desc = f'Epoch {epoch}/{config["train"]["num_epochs"]}',
        )

        train_loss_avg = 0
        for _, (images, labels) in enumerate(local_progress):  

            model.zero_grad()
            
            outputs = model.forward(
                images.to(device, non_blocking=True),
            )

            labels = labels.float().to(device, non_blocking=True)
            
            # Compute loss for each batch
            batch_loss = criterion(outputs, labels).mean()
            
            # Backward pass and optimize
            batch_loss.backward()
            optimizer.step()

            train_loss_avg += batch_loss

        # Number of batch in training set
        train_num_batch = train_size / config["train"]["batch_size"]
        # Average the total loss of all batches
        train_loss_avg = train_loss_avg / train_num_batch

        # TODO: i think you should move this after the eval section below. check TODO (*)
        model_save_path = f'{config["output_folder"]}/epoch_{epoch}_trainloss_{train_loss_avg:.6f}.pt'
        torch.save(model, model_save_path)

        print(f'Training loss: {train_loss_avg:.6f}')


        with torch.no_grad():
            model.eval()

            val_loss_avg = 0
            for _, (images, labels) in enumerate(val_data_loader):  
                
                outputs = model.forward(
                    images.to(device, non_blocking=True),
                )

                labels = labels.float().to(device, non_blocking=True)
                
                # Compute loss for each batch
                batch_loss = criterion(outputs, labels).mean()

                val_loss_avg += batch_loss

            # Number of batch in validation set
            val_num_batch = val_size / config["train"]["batch_size"]
            # Average the total loss of all batches
            val_loss_avg = val_loss_avg / val_num_batch

            print(f'Validation loss: {val_loss_avg:.6f}')

        # TODO (*): i think you should put the model saving part here, with an if condition:
        #           if val_loss_avg is smaller than the lowest val loss ever seen, then
        #           (1) save the model
        #           (2) update the lowest val loss = val_loss_avg
        #           see https://github.com/kuanweih/strong_lensing_vit_resnet/blob/4392b4e328e3ccc81160b11da7b082da7b80f322/train_model.py#L251

        # TODO for both train and validation section, in addition to recording the loss, i think we also
        # need to record the following quantities because we are interested in recall and precision:
        # (1) number of samples with correct prediction
        # (2) number of total samples
        # (3) number of positive samples with correct prediction
        # (4) number of negative samples with correct prediction
        # (5) number of total positive samples
        # (6) number of total negative samples
        # these are the ones on top of my head. feel free to adjust it with you judguement!












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

    num_gpus = torch.cuda.device_count()        ####
    print(f"Number of used GPUs: {num_gpus}")   ####

    main(device=args.device, config=config)

    # # Wrap up logs
    # completed_output_dir = args.output_dir.replace('in-progress', 'completed')
    # os.rename(args.output_dir, completed_output_dir)
    # print(f'Output has been saved to {completed_output_dir}')

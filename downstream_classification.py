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

import matplotlib.pyplot as plt 
import pandas as pd

def main(device, config):

    # Load dataset
    dataset = get_hst_x_zoo(
        config["dataset"]["data_dir"],
        subset_size=config["dataset"]["subset_size"],
    )

    train_size = int(config["train"]["train_fraction"] * len(dataset))
    test_size = len(dataset) - train_size

    # Set the random seed
    torch.manual_seed(42)

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])

    # Create data loaders for train, val, and test sets
    train_data_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        shuffle = True,
        batch_size = config["train"]["batch_size"],
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        shuffle = False,
        batch_size = config["train"]["batch_size"], 
    )


    # Load the trained backbone model
    model = get_backbone(config["model"]["backbone"]).to(device)

    # fill in pre-trained weights if provided
    if config["model"]["load_trained_weights"]:
        file_path = os.path.join(config["model"]["file"])
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist!")
        else:
            ckpt = torch.load(file_path, map_location='cpu')
            assert ckpt["backbone"] == config["model"]["backbone"]  # make sure loaded model == model
            model.load_state_dict(ckpt["backbone_state_dict"])



    # # Freeze parameters of the pre-trained model  
    # for param in model.parameters():
    #     param.requires_grad = False


    # Define the last layer
    # Output dimension of the last layer
    output_dim = 2  # Binary classification (0: not lens. 1: lens)

    if config["model"]["backbone"] == 'vit-base':

        # Input dimension of the last layer
        input_dim = model.encoder.ln.normalized_shape[0]  # This will be 768 from our setting

        model.heads = nn.Sequential(
        nn.Linear(input_dim, output_dim),  
        # nn.Sigmoid(),  # Use BCEWithLogitsLoss so no need for sigmoid here
        )    

    else: # resnet

        # Input dimension of the last layer
        input_dim = model.layer4[-1].bn3.num_features  # This will be 2048 for ResNet-101

        model.fc = nn.Sequential(
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
    # Move pos_weight to the device
    pos_weight = pos_weight.to(device) 
    criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)


# -------------------------


    global_progress = tqdm(range(0, config["train"]["num_epochs"]), desc='Training')

    epoch_list = []
    train_loss_list = []
    test_loss_list = []
    
    train_label_pred_epoch_list = []
    test_label_pred_epoch_list = []
    

    for epoch in global_progress:

        model.train()

        train_local_progress = tqdm(
            train_data_loader,
            desc = f'Epoch {epoch}/{config["train"]["num_epochs"]}',
        )

        train_loss_avg = 0
        train_pred_epoch = torch.empty((0, 2)).to(device)
        if epoch == 0:
            train_label = torch.empty((0, 2)).to(device)


        for _, (images, labels) in enumerate(train_local_progress):  

            model.zero_grad()
            
            outputs = model(
                images.to(device, non_blocking=True),  # outputs: one-hot format;  dim = (batch_size, 2)
            )
            print(outputs)

            labels = labels.float().to(device, non_blocking=True)  # labels: one-hot format;  dim = (batch_size, 2)
            
            # Compute loss for each batch
            batch_loss = criterion(outputs, labels).mean()
            train_loss_avg += batch_loss

            # Backward pass and optimize
            batch_loss.backward()
            optimizer.step()

            # m=nn.Sigmoid()
            m = nn.Softmax(dim=1)
            outputs = m(outputs) # dim = (batch_size, 2)


            train_pred_epoch = torch.cat((train_pred_epoch, outputs), dim=0)
            if epoch == 0:
                train_label = torch.cat((train_label, labels), dim=0)


        # Number of batch in training set
        train_num_batch = train_size / config["train"]["batch_size"]
        # Average the total loss of all batches
        train_loss_avg = train_loss_avg / train_num_batch
        # print(f'Training loss: {train_loss_avg:.6f}')

        if epoch == 0:
            train_label_pred_epoch_list.append(train_label.detach().cpu().numpy())
        train_label_pred_epoch_list.append(train_pred_epoch.detach().cpu().numpy()) 
        np.save(f'{config["output_folder"]}/train_label_pred_epoch.npy', train_label_pred_epoch_list)

# -------------------------

        # Test the model on test set
        test_local_progress = tqdm(
            test_data_loader,
            desc = f'Epoch {epoch}/{config["train"]["num_epochs"]}',
        )

        with torch.no_grad():
            model.eval()

            test_loss_avg = 0
            test_pred_epoch = torch.empty((0, 2)).to(device)
            if epoch == 0:
                test_label = torch.empty((0, 2)).to(device)

            for _, (images, labels) in enumerate(test_local_progress):  
                
                outputs = model(
                    images.to(device, non_blocking=True),    # outputs: one-hot format;  dim = (batch_size, 2)
                )

                labels = labels.float().to(device, non_blocking=True)   # labels: one-hot format;  dim = (batch_size, 2)       

                # Compute loss for each batch
                batch_loss = criterion(outputs, labels).mean()
                test_loss_avg += batch_loss        
 
                # m=nn.Sigmoid()
                m = nn.Softmax(dim=1)
                outputs = m(outputs) # dim = (batch_size, 2)

                test_pred_epoch = torch.cat((test_pred_epoch, outputs), dim=0)
                if epoch == 0:
                    test_label = torch.cat((test_label, labels), dim=0)

            # Number of batch in test set
            test_num_batch = test_size / config["train"]["batch_size"]
            # Average the total loss of all batches
            test_loss_avg = test_loss_avg / test_num_batch
            # print(f'Test loss: {test_loss_avg:.6f}')

            if epoch == 0:
                test_label_pred_epoch_list.append(test_label.detach().cpu().numpy())
            test_label_pred_epoch_list.append(test_pred_epoch.detach().cpu().numpy()) 
            np.save(f'{config["output_folder"]}/test_label_pred_epoch.npy', test_label_pred_epoch_list)

            
        # TODO (*): i think you should put the model saving part here, with an if condition:
        #           if test_loss_avg is smaller than the lowest val loss ever seen, then
        #           (1) save the model
        #           (2) update the lowest test loss = test_loss_avg
        #           see https://github.com/kuanweih/strong_lensing_vit_resnet/blob/4392b4e328e3ccc81160b11da7b082da7b80f322/train_model.py#L251

        
 
        # Record all losses, make plot, and save the outputs
        epoch_list.append(epoch)
        train_loss_list.append(train_loss_avg.detach().cpu().numpy())
        test_loss_list.append(test_loss_avg.detach().cpu().numpy())

        plt.plot(epoch_list, train_loss_list, '-o', color= 'navy')
        plt.plot(epoch_list, test_loss_list, '-o', color= 'coral')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(f'{config["output_folder"]}/plotter.pdf')
        

        output_dict = {
            'Epoch': epoch_list,
            'Train_loss': train_loss_list,
            'Test_loss': test_loss_list,
        }
        df = pd.DataFrame(output_dict)
        df.to_csv(f'{config["output_folder"]}/output_history.csv', index=False)



        if test_loss_avg < train_loss_avg:
            lowest_loss = test_loss_avg
            model_save_path = f'{config["output_folder"]}/epoch_{epoch}_loss_{lowest_loss:.6f}.pt'
            torch.save(model, model_save_path)






        # TODO for both train and validation section, in addition to recording the loss, i think we also
        # need to record the following quantities because we are interested in recall and precision:
        # (1) number of samples with correct prediction TP+TN
        # (2) number of total samples TP+FP+TN+FN
        # (3) number of positive samples with correct prediction TP
        # (4) number of negative samples with correct prediction TN
        # (5) number of total positive samples TP+FN
        # (6) number of total negative samples TN+FP
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

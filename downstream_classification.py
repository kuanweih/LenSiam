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
    val_size = int(config["train"]["val_fraction"] * len(dataset))
    test_size = len(dataset) - (train_size + val_size)

    # Set the random seed
    torch.manual_seed(42)

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
    # ckpt = torch.load(config["model"]["file"], map_location='cpu')
    # assert ckpt["backbone"] == config["model"]["backbone"]  # make sure loaded model == model
    # model.load_state_dict(ckpt["backbone_state_dict"])


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
    # Move pos_weight to the device
    pos_weight = pos_weight.to(device) 
    criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)


# -------------------------


    global_progress = tqdm(range(0, config["train"]["num_epochs"]), desc='Training')

    epoch_list = []
    train_loss_list = []
    val_loss_list = []

    train_TP_list = []
    train_FP_list = []
    train_TN_list = []
    train_FN_list = []

    val_TP_list = []
    val_FP_list = []
    val_TN_list = []
    val_FN_list = []

    test_TP_list = []
    test_FP_list = []
    test_TN_list = []
    test_FN_list = []

    # train_accuracy_list = []
    # train_precision_list = []
    # train_recall_list = []
    # train_f1_list = []

    # val_accuracy_list = []
    # val_precision_list = []
    # val_recall_list = []
    # val_f1_list = []

    # test_accuracy_list = []
    # test_precision_list = []
    # test_recall_list = []
    # test_f1_list = []

    for epoch in global_progress:

        model.train()

        train_local_progress = tqdm(
            train_data_loader,
            desc = f'Epoch {epoch}/{config["train"]["num_epochs"]}',
        )

        train_loss_avg = 0
        train_TP = 0
        train_FP = 0
        train_TN = 0
        train_FN = 0
        train_T = 0
        train_predictions = 0

        for _, (images, labels) in enumerate(train_local_progress):  

            model.zero_grad()
            
            outputs = model(
                images.to(device, non_blocking=True),  # outputs: one-hot format
            )

            labels = labels.float().to(device, non_blocking=True)  # labels: one-hot format
            
            # Compute loss for each batch
            batch_loss = criterion(outputs, labels).mean()
            
            # Backward pass and optimize
            batch_loss.backward()
            optimizer.step()

            train_loss_avg += batch_loss


            with torch.no_grad():
                model.eval()

                outputs = model(
                    images.to(device, non_blocking=True),
                )

                m=nn.Sigmoid()
                outputs = m(outputs)
                # Convert the one-hot outputs to a prediction tensor with only 0, 1
                train_predictions = torch.argmax(outputs, dim=1)  
                
                # Convert the one-hot labels to a prediction tensor with only 0, 1
                train_labels = torch.argmax(labels, dim=1)

                train_TP += ((train_predictions == 1) & (train_labels == 1)).sum().item()
                train_FP += ((train_predictions == 1) & (train_labels == 0)).sum().item()
                train_TN += ((train_predictions == 0) & (train_labels == 0)).sum().item()
                train_FN += ((train_predictions == 0) & (train_labels == 1)).sum().item()
                train_T += torch.eq(train_predictions, train_labels).sum().item()
                train_predictions += len(train_labels.detach().cpu().numpy())
            
            model.train()


        # Number of batch in training set
        train_num_batch = train_size / config["train"]["batch_size"]
        # Average the total loss of all batches
        train_loss_avg = train_loss_avg / train_num_batch
        # print(f'Training loss: {train_loss_avg:.6f}')

        
# -------------------------


        # Test the model on validation set
        val_local_progress = tqdm(
            val_data_loader,
            desc = f'Epoch {epoch}/{config["train"]["num_epochs"]}',
        )

        with torch.no_grad():
            model.eval()

            val_loss_avg = 0
            val_TP = 0
            val_FP = 0
            val_TN = 0
            val_FN = 0
            val_T = 0
            val_predictions = 0

            for _, (images, labels) in enumerate(val_local_progress):  
                
                outputs = model(
                    images.to(device, non_blocking=True),
                )

                labels = labels.float().to(device, non_blocking=True)
                
                # Compute loss for each batch
                batch_loss = criterion(outputs, labels).mean()

                val_loss_avg += batch_loss
                
 
                m=nn.Sigmoid()
                outputs = m(outputs)
                # Convert the one-hot outputs to a prediction tensor with only 0, 1
                val_predictions = torch.argmax(outputs, dim=1)  
                
                # Convert the one-hot labels to a prediction tensor with only 0, 1
                val_labels = torch.argmax(labels, dim=1)

                val_TP += ((val_predictions == 1) & (val_labels == 1)).sum().item()
                val_FP += ((val_predictions == 1) & (val_labels == 0)).sum().item()
                val_TN += ((val_predictions == 0) & (val_labels == 0)).sum().item()
                val_FN += ((val_predictions == 0) & (val_labels == 1)).sum().item()
                val_T += torch.eq(val_predictions, val_labels).sum().item()
                val_predictions += len(val_labels.detach().cpu().numpy())


            # Number of batch in validation set
            val_num_batch = val_size / config["train"]["batch_size"]
            # Average the total loss of all batches
            val_loss_avg = val_loss_avg / val_num_batch
            # print(f'Validation loss: {val_loss_avg:.6f}')

# -------------------------

        # Test the model on testing set
        test_local_progress = tqdm(
            test_data_loader,
            desc = f'Epoch {epoch}/{config["train"]["num_epochs"]}',
        )

        with torch.no_grad():
            model.eval()

            test_TP = 0
            test_FP = 0
            test_TN = 0
            test_FN = 0
            test_T = 0
            test_predictions = 0

            for _, (images, labels) in enumerate(test_local_progress):  
                
                outputs = model(
                    images.to(device, non_blocking=True),
                )

                labels = labels.float().to(device, non_blocking=True)                
 
                m=nn.Sigmoid()
                outputs = m(outputs)
                # Convert the one-hot outputs to a prediction tensor with only 0, 1
                test_predictions = torch.argmax(outputs, dim=1)  
                
                # Convert the one-hot labels to a prediction tensor with only 0, 1
                test_labels = torch.argmax(labels, dim=1)

                test_TP += ((test_predictions == 1) & (test_labels == 1)).sum().item()
                test_FP += ((test_predictions == 1) & (test_labels == 0)).sum().item()
                test_TN += ((test_predictions == 0) & (test_labels == 0)).sum().item()
                test_FN += ((test_predictions == 0) & (test_labels == 1)).sum().item()
                test_T += torch.eq(test_predictions, test_labels).sum().item()
                test_predictions += len(test_labels.detach().cpu().numpy())

            
        # TODO (*): i think you should put the model saving part here, with an if condition:
        #           if val_loss_avg is smaller than the lowest val loss ever seen, then
        #           (1) save the model
        #           (2) update the lowest val loss = val_loss_avg
        #           see https://github.com/kuanweih/strong_lensing_vit_resnet/blob/4392b4e328e3ccc81160b11da7b082da7b80f322/train_model.py#L251

        
 
        # Record all losses, make plot, and save the outputs
        epoch_list.append(epoch)
        train_loss_list.append(train_loss_avg.detach().cpu().numpy())
        val_loss_list.append(val_loss_avg.detach().cpu().numpy())

        plt.plot(epoch_list, train_loss_list, '-o', color= 'navy', label = 'Train')
        plt.plot(epoch_list, val_loss_list, '-o', color= 'coral', label = 'Val')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{config["output_folder"]}/plotter.pdf')
        

        train_TP_list.append(train_TP)
        train_FP_list.append(train_FP)
        train_TN_list.append(train_TN)
        train_FN_list.append(train_FN)

        val_TP_list.append(val_TP)
        val_FP_list.append(val_FP)
        val_TN_list.append(val_TN)
        val_FN_list.append(val_FN)

        test_TP_list.append(test_TP)
        test_FP_list.append(test_FP)
        test_TN_list.append(test_TN)
        test_FN_list.append(test_FN)

        # def accuracy(T, total):
        #     return T/total
        # def precision(TP, FP):
        #     return TP/(TP+FP)
        # def recall(TP, FN):
        #     return TP/(TP+FN)
        # def f1(TP, FP, FN):
        #     return 2*(precision(TP, FP)*recall(TP, FN))/ (precision(TP, FP)+recall(TP, FN))

        # train_accuracy_list.append(accuracy(train_T, train_predictions))
        # train_precision_list.append(precision(train_TP, train_FP))
        # train_recall_list.append(recall(train_TP, train_FN))
        # train_f1_list.append(f1(train_TP, train_FP, train_FN))

        # val_accuracy_list.append(accuracy(val_T, val_predictions))
        # val_precision_list.append(precision(val_TP, val_FP))
        # val_recall_list.append(recall(val_TP, val_FN))
        # val_f1_list.append(f1(val_TP, val_FP, val_FN))

        # test_accuracy_list.append(accuracy(test_T, test_predictions))
        # test_precision_list.append(precision(test_TP, test_FP))
        # test_recall_list.append(recall(test_TP, test_FN))
        # test_f1_list.append(f1(test_TP, test_FP, test_FN))


        # output_dict = {
        #     'Epoch': epoch_list,
        #     'Train_loss': train_loss_list,
        #     'Val_loss': val_loss_list,
        #     'Train_acc': train_accuracy_list,
        #     'Train_prec': train_precision_list,
        #     'Train_rec': train_recall_list,
        #     'Train_f1': train_f1_list,
        #     'Val_acc': val_accuracy_list,
        #     'Val_prec': val_precision_list,
        #     'Val_rec': val_recall_list,
        #     'Val_f1': val_f1_list,
        #     'Test_acc': test_accuracy_list,
        #     'Test_prec': test_precision_list,
        #     'Test_rec': test_recall_list,
        #     'Test_f1': test_f1_list,
        # }
        output_dict = {
            'Epoch': epoch_list,
            'Train_loss': train_loss_list,
            'Val_loss': val_loss_list,
            'Train_TP': train_TP_list,
            'Train_FP': train_FP_list,
            'Train_TN': train_TN_list,
            'Train_FN': train_FN_list,
            'Val_TP': val_TP_list,
            'Val_FP': val_FP_list,
            'Val_TN': val_TN_list,
            'Val_FN': val_FN_list,
            'Test_TP': test_TP_list,
            'Test_FP': test_FP_list,
            'Test_TN': test_TN_list,
            'Test_FN': test_FN_list,
        }
        df = pd.DataFrame(output_dict)
        df.to_csv(f'{config["output_folder"]}/output_history.csv', index=False)



        if val_loss_avg < train_loss_avg:
            lowest_loss = val_loss_avg
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

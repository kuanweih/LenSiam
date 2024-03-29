import os
import torch

from tqdm import tqdm
from arguments import get_args
from datasets import get_dataset
from models import get_model
from optimizers import get_optimizer, LR_Scheduler
from tools import Logger, Checkpointer


def main(device, args):
    
    args.dataset_kwargs['aug_method'] = args.aug_method
    # Load dataset
    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(**args.dataset_kwargs),
        shuffle=True,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs,
    )

    # Initialize the model, loss function, and optimizer
    model = get_model(args.model).to(device)
    model = torch.nn.DataParallel(model)
    optimizer = get_optimizer(model, args)
    lr_scheduler = LR_Scheduler(optimizer, len(train_loader), args, constant_predictor_lr=True)
    # constant_predictor_lr=True: see the end of section 4.2 predictor

    # Initialize helper tools
    logger = Logger(log_dir=args.log_dir, matplotlib=args.logger.matplotlib)
    checkpointer = Checkpointer(args)
    lowest_loss = float("inf")  # to identify model with the lowest loss

    # Train the model
    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc='Training')
    for epoch in global_progress:
        model.train()

        local_progress = tqdm(
            train_loader,
            desc=f'Epoch {epoch}/{args.train.num_epochs}',
        )
        for (images1, images2, labels, paths) in local_progress:
            # labels and paths will not be used

            model.zero_grad()
            data_dict = model(
                images1.to(device, non_blocking=True),
                images2.to(device, non_blocking=True),
            )
            loss = data_dict['loss'].mean() # ddp
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            data_dict.update({'lr': lr_scheduler.get_lr()})
            local_progress.set_postfix(data_dict)
            logger.update_scalers(data_dict)

            if loss.item() < lowest_loss:
                lowest_loss = loss.item()
                checkpointer.save(epoch, model, loss.item())
        
        # logs, plots, ckpts
        epoch_dict = {
            "epoch": epoch,
            "loss": loss.item(),
        }
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)

        if (epoch + 1) % args.train.n_epoch_ckpt == 0:
            checkpointer.save(epoch, model, loss.item())


if __name__ == "__main__":

    args = get_args()
    print(f"We will be using device = {args.device}!")
    
    # Show number of GPUs in use
    num_gpus = torch.cuda.device_count()        
    print(f"Number of used GPUs: {num_gpus}")   
    
    main(device=args.device, args=args)

    # wrap up logs
    completed_log_dir = args.log_dir.replace('in-progress', 'completed')
    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')















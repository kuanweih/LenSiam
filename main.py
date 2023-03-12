import os
import torch

from datetime import datetime
from tqdm import tqdm

from arguments import get_args
from datasets import get_dataset
from models import get_model
from optimizers import get_optimizer, LR_Scheduler
from tools import Logger


def main(device, args):

    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(**args.dataset_kwargs),
        shuffle=True,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs,
    )

    model = get_model(args.model).to(device)
    model = torch.nn.DataParallel(model)

    # define optimizer
    optimizer = get_optimizer(
        args.train.optimizer.name,
        model,
        lr=args.train.base_lr*args.train.batch_size/256,
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay,
    )

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs,
        args.train.warmup_lr * args.train.batch_size / 256,
        args.train.num_epochs,
        args.train.base_lr * args.train.batch_size / 256,
        args.train.final_lr * args.train.batch_size / 256,
        len(train_loader),
        constant_predictor_lr=True, # see the end of section 4.2 predictor
    )

    logger = Logger(
        log_dir=args.log_dir,
        matplotlib=args.logger.matplotlib,
    )

    # Start training
    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc='Training')
    for epoch in global_progress:
        model.train()

        local_progress = tqdm(
            train_loader,
            desc=f'Epoch {epoch}/{args.train.num_epochs}',
        )
        for idx, (images1, images2, labels) in enumerate(local_progress):
            # idx and labels will not be used

            model.zero_grad()
            data_dict = model.forward(
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

        epoch_dict = {
            "epoch": epoch,
            "loss": loss.item(),
        }
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)

    # Save checkpoint
    model_file_name = f"{args.name}_{datetime.now().strftime('%m%d%H%M%S')}.pth"
    model_path = os.path.join(args.ckpt_dir, model_file_name)
    torch.save(
        {
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
        },
        model_path,
    )
    print(f"Model saved to {model_path}")
    with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as file:
        file.write(f'{model_path}')




if __name__ == "__main__":

    args = get_args()
    print(f"We will be using device = {args.device}!")

    main(device=args.device, args=args)

    # wrap up logs
    completed_log_dir = args.log_dir.replace('in-progress', 'completed')
    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')














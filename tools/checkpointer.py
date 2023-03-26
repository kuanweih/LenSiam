import os
import torch

from datetime import datetime


class Checkpointer:

    def __init__(self, args):
        self.args = args

    def save(self, epoch, model):
        model_file_name = f"{self.args.name}_{datetime.now().strftime('%m%d%H%M%S')}.pt"
        model_path = os.path.join(self.args.ckpt_dir, model_file_name)
        torch.save(
            {
                'epoch': epoch + 1,
                'full_model': self.args.model.name,            
                'full_model_state_dict': model.module.state_dict(),  # entire simsiam model
                'backbone': self.args.model.backbone,
                'backbone_state_dict': model.module.backbone.state_dict(),  # backbone model
            },
            model_path,
        )
        print(f"Model saved to {model_path}")
        with open(os.path.join(self.args.log_dir, f"checkpoint_path.txt"), 'a') as file:
            file.write(f'{model_path}\n')


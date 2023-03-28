from .lars import LARS
import torch
from .lr_scheduler import LR_Scheduler


def get_optimizer(model, args):

    # get variables from args
    optimizer_name = args.optimizer.name
    lr = args.train.base_lr * args.train.batch_size / 256
    momentum = args.optimizer.momentum
    weight_decay = args.optimizer.weight_decay

    predictor_prefix = ('module.predictor', 'predictor')
    parameters = [
        {
            'name': 'base',
            'params': [param for name, param in model.named_parameters()
                       if not name.startswith(predictor_prefix)],
            'lr': lr,
        },
        {
            'name': 'predictor',
            'params': [param for name, param in model.named_parameters()
                       if name.startswith(predictor_prefix)],
            'lr': lr,
        },
    ]

    if optimizer_name == 'lars':
        optimizer = LARS(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer




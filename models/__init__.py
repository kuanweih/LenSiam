import os
import torch

from torchvision.models import resnet50, resnet18, vit_b_16
from .simsiam import SimSiam


def get_backbone(backbone, castrate=True):
    """ Get backbone model according to the argument backbone.
    Args:
        backbone (str): name of the backbone model
        castrate (bool, optional): replace the last layer of backbone with torch.nn.Identity().
                                   Defaults to True.
    Returns:
        torch model: the backbone model
    """
    if backbone == 'resnet18':
        backbone_model = resnet18()
    elif backbone == 'resnet50':
        backbone_model = resnet50()
    elif backbone == 'vit-base':
        backbone_model = vit_b_16()
    else:
        raise NotImplementedError

    if castrate:
        if backbone in ["resnet18", "resnet50"]:
            backbone_model.output_dim = backbone_model.fc.in_features
            backbone_model.fc = torch.nn.Identity()
        elif backbone in ["vit-base"]:
            backbone_model.output_dim = backbone_model.heads.head.in_features
            backbone_model.heads.head = torch.nn.Identity()
        else:
            raise NotImplementedError
    return backbone_model


def get_model(model_cfg):
    """ Get the main model, e.g. SimSiam = backbone + projector.
    Args:
        model_cfg (arguments.Namespace): model config
    Returns:
        torch model: the main model
    """
    # get model
    if model_cfg.name == 'simsiam':
        model =  SimSiam(get_backbone(model_cfg.backbone))
    else:
        raise NotImplementedError

    # fill in pre-trained weights if provided
    if model_cfg.load_trained_weights:
        file_path = os.path.join(model_cfg.trained_weights_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist!")
        else:
            ckpt = torch.load(file_path)
            assert ckpt["full_model"] == model_cfg.name  # make sure loaded model == model
            model.load_state_dict(ckpt["full_model_state_dict"])
    return model

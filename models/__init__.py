import torch
from torchvision.models import resnet50, resnet18, vit_b_16
from .simsiam import SimSiam


def get_backbone(backbone, castrate=True):
    """ Get backbone model according to the argument backbone.
    Args:
        backbone (str): name of the backbone model
        castrate (bool, optional): [description]. Defaults to True.
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
    if model_cfg.name == 'simsiam':
        model =  SimSiam(get_backbone(model_cfg.backbone))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    else:
        raise NotImplementedError
    return model

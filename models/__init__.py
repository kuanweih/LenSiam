import torch
from torchvision.models import resnet50, resnet18
from .simsiam import SimSiam


def get_backbone(backbone, castrate=True):
    """ Get backbone model according to the argument backbone.
    Args:
        backbone (str): name of the backbone model
        castrate (bool, optional): [description]. Defaults to True.
    Returns:
        torch model: the backbone model
    """
    # TODO: add bigger resnet and vit backbone here!

    if backbone == 'resnet18':
        backbone_model = resnet18()
    else:
        raise NotImplementedError

    # TODO check paper to see if this is neccesary?
    if castrate:
        backbone_model.output_dim = backbone_model.fc.in_features
        backbone_model.fc = torch.nn.Identity()

    return backbone_model


def get_model(model_cfg):
    """ Get the main model, e.g. SimSiam = backbone + projector.
    Args:
        model_cfg (?): model config TODO make it dict?
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







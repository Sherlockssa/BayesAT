

from .resnet import resnet, Normalization
from .wideresnet import wideresnet

def create_model(args, info, device, logger):
    name, normalize = args.model, args.normalize
    if 'resnet' in name:
        backbone = resnet(name, num_classes=info['num_classes'], device=device, normalize=normalize, mean=info['mean'], std=info['std'])
    elif 'wrn' in name:
        backbone = wideresnet(name, logger, num_classes=info['num_classes'], device=device, normalize=normalize, mean=info['mean'], std=info['std'])
    else:
        raise ValueError('Invalid model name {}!'.format(name))
    backbone = backbone.to(device)
    args.feat_dim = backbone.feat_dim
    return backbone


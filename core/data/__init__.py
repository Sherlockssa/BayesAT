import os

import numpy as np
import torch
import torchvision

from .cifar10 import load_cifar10
from .cifar100 import load_cifar100
from .imagenet32 import load_imagenet32
from .imagenet64 import load_imagenet64
from .sampler import get_semisup_dataloaders

DATASETS = []
_LOAD_DATASET_FN = {
    'cifar10': load_cifar10,
    'cifar100': load_cifar100,
    'imagenet32': load_imagenet32,
    'imagenet64': load_imagenet64,
}
for d in _LOAD_DATASET_FN:
    DATASETS.append(d)

def get_data_info(data_dir):
    """
    Returns dataset information.
    Arguments:
        data_dir (str): path to data directory.
    """
    dataset = os.path.basename(os.path.normpath(data_dir))
    if 'cifar100' in data_dir:
        from .cifar100 import DATA_DESC
    elif 'cifar10' in data_dir:
        from .cifar10 import DATA_DESC
    elif 'imagenet32' in data_dir:
        from .imagenet32 import DATA_DESC
    elif 'imagenet64' in data_dir:
        from .imagenet64 import DATA_DESC
    else:
        raise ValueError(f'Only data in {DATASETS} are supported!')
    DATA_DESC['data'] = dataset
    return DATA_DESC


def load_data(data_dir, logger, batch_size=256, batch_size_test=256, num_workers=4, use_augmentation='none', use_consistency=False,
              aux_data_filename=None, unsup_fraction=None, validation=False, take_amount=None, aux_take_amount=None, take_amount_seed=1, add_aux_labels=False,
              pseudo_label_model=None, num_batches=None):

    dataset = os.path.basename(os.path.normpath(data_dir))
    load_dataset_fn = _LOAD_DATASET_FN[dataset]

    logger.log(f'Dataset {dataset} (seed:{take_amount_seed}, take_amount:{take_amount}) is loading.')
    train_dataset, test_dataset, val_dataset = load_dataset_fn(data_dir=data_dir,
                                                               use_augmentation=use_augmentation,
                                                               use_consistency=use_consistency,
                                                               validation=validation,
                                                               take_amount=take_amount,
                                                               aux_take_amount=aux_take_amount,
                                                               aux_data_filename=aux_data_filename,
                                                               take_amount_seed=take_amount_seed,
                                                               add_aux_labels=add_aux_labels,
                                                               pseudo_label_model=pseudo_label_model,
                                                               logger=logger
                                                               )

    if num_batches is None:
        dataset_size = train_dataset.origen_len
        if unsup_fraction <= 0:
            dataset_size = len(train_dataset.sup_indices)
        num_batches = int(np.ceil(dataset_size / batch_size))
        print(f">>> {dataset_size} / {batch_size} = {num_batches}")
    if validation:
        train_dataloader, test_dataloader, val_dataloader = get_semisup_dataloaders(
            train_dataset, test_dataset, val_dataset,
            batch_size=batch_size, batch_size_test=batch_size_test,
            num_workers=num_workers, unsup_fraction=unsup_fraction,
            num_batches=num_batches,
            logger=logger
        )
        return train_dataset, test_dataset, val_dataset, train_dataloader, test_dataloader, val_dataloader
    else:
        train_dataloader, test_dataloader = get_semisup_dataloaders(
            train_dataset, test_dataset, None, batch_size=batch_size, batch_size_test=batch_size_test,
            num_workers=num_workers, unsup_fraction=unsup_fraction,
            num_batches=num_batches,
            logger=logger
        )
        return train_dataset, test_dataset, None, train_dataloader, test_dataloader, None


def load_test_data(data_dir, batch_size_test=256, num_workers=4, shuffle=False):
    dataset = os.path.basename(os.path.normpath(data_dir))
    test_transform = torchvision.transforms.ToTensor()

    if 'cifar100' in data_dir:
        test_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)

    elif 'cifar10' in data_dir:
        data_dir = os.path.join(os.path.dirname(data_dir), 'cifar10')
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    elif 'imagenet32' in data_dir:
        from .imagenet32 import ImageNet32
        test_dataset = ImageNet32(root=data_dir, train=False, transform=test_transform)
    elif 'imagenet64' in data_dir:
        from .imagenet64 import ImageNet64
        test_dataset = ImageNet64(root=data_dir, train=False, transform=test_transform)
    else:
        raise ValueError(f'Only data in {DATASETS} are supported!')

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=shuffle,
                                                  num_workers=num_workers)

    return test_dataset, test_dataloader

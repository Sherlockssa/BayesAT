import os
import shutil
import urllib
import zipfile

import torch
import os.path
from typing import Any, Callable, Optional, Tuple
import numpy as np
from PIL import Image
from torchvision.transforms import AutoAugmentPolicy
from tqdm import tqdm

try: from torchvision.transforms import v2 as T
except ImportError: import torchvision.transforms as T
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets import ImageFolder
from .dataset import SemiSupervisedDataset


NUM_CLASSES = 200
DATA_DESC = {
    'data': 'imagenet',
    'classes': tuple(range(0, NUM_CLASSES)),
    'num_classes': NUM_CLASSES,
    'mean': (0.485, 0.456, 0.406),
    'std': (0.229, 0.224, 0.225),
}

class Imagenet(VisionDataset):
    # http://image-net.org/download-images
    # Imagenet64, a Downsampled Variant of ImageNet
    # https://arxiv.org/pdf/1707.08819.pdf

    train_list = [
        'train_data_batch_1.npz',
        'train_data_batch_2.npz',
        'train_data_batch_3.npz',
        'train_data_batch_4.npz',
        'train_data_batch_5.npz',
        'train_data_batch_6.npz',
        'train_data_batch_7.npz',
        'train_data_batch_8.npz',
        'train_data_batch_9.npz',
        'train_data_batch_10.npz',
    ]
    valid_list = [
        'val_data.npz',
    ]

    def __init__(
            self,
            root: str = '../dataset_data/imagenet/',
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download=True, # download from http://image-net.org/download-images
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.valid_list
        self.data = []
        self.targets = []

        print(f"preparing ...")
        for i, file_name in enumerate(downloaded_list):
            print(file_name)
            file_path = os.path.join(self.root, file_name)
            entry = np.load(file_path)
            self.data.append(entry["data"])
            self.targets.extend(entry["labels"])
        self.data = np.vstack(self.data).reshape(-1, 3, 64, 64)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        num_classes = NUM_CLASSES
        # print(f"subsampling {num_classes} classes ... from {set(self.targets)}")
        if num_classes < 1000:
            selected_indices = [i for i, label in enumerate(self.targets) if label <= num_classes]
            self.data = self.data[selected_indices]
            self.targets = np.array([self.targets[i] - 1 for i in selected_indices])
        # raise NotImplementedError

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)


class Imagenet_LT(SemiSupervisedDataset):
    def load_base_dataset(self, **kwargs):
        base_dataset = DATA_DESC['data']
        assert self.base_dataset == base_dataset, f'Only imbalanced {base_dataset} is supported. Please use correct dataset!'
        self.dataset = Imagenet(train=True, **kwargs)
        self.dataset.num_classes = DATA_DESC['num_classes']
        self.dataset.mean_std = (DATA_DESC['mean'], DATA_DESC['std'])

def load_imagenetlf(
        data_dir, logger, validation=False,
        imb_rate=10, imb_type='exp', seed=1,
        label_fraction=1., augmentation='none',
):
    base_dataset = DATA_DESC['data']
    data_dir = os.path.join(os.path.dirname(data_dir), base_dataset)
    data_args = dict(root=data_dir, download=True)

    if augmentation == 'base':
        transform_train = T.Compose([
            T.RandomCrop(64, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor()]
        )
    elif augmentation == 'aua':
        transform_aug = [T.AutoAugment(policy=AutoAugmentPolicy.CIFAR10)]
        transform_train = T.Compose(transform_aug + [
            T.RandomCrop(64, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
    elif augmentation == 'ra':
        transform_aug = [T.RandAugment(2,8)]
        transform_train = T.Compose(transform_aug + [
            T.RandomCrop(64, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
    elif augmentation == 'none':
        transform_train = T.Compose([T.ToTensor()])
    else:
        raise ValueError('Unknown augmentation {}'.format(augmentation))

    train_dataset = Imagenet_LT(
        base_dataset=base_dataset, validation=validation,
        imb_rate=imb_rate, imb_type=imb_type, seed=seed, label_fraction=label_fraction,
        transform=transform_train, logger=logger, **data_args,
    )
    if validation:
        transform_test = T.Compose([T.ToTensor()])
        val_dataset = Imagenet(**data_args, train=True, transform=transform_test)
        val_dataset = torch.utils.data.Subset(val_dataset, train_dataset.val_indices)
        return train_dataset, val_dataset
    return train_dataset, None


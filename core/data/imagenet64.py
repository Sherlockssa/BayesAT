import os
import os.path
from typing import Callable, Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets.vision import VisionDataset

from .dataset import SemiSupervisedDataset

DATA_DESC = {
    'data': 'imagenet',
    'classes': tuple(range(0, 200)),
    'num_classes': 200,
    'mean': (0.485, 0.456, 0.406),
    'std': (0.229, 0.224, 0.225),
}

class ImageNet64(VisionDataset):
    # http://image-net.org/download-images
    # A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets
    # https://arxiv.org/pdf/1707.08819.pdf

    train_list = [
        'train_data_batch_1',
        'train_data_batch_2',
        'train_data_batch_3',
        'train_data_batch_4',
        'train_data_batch_5',
        'train_data_batch_6',
        'train_data_batch_7',
        'train_data_batch_8',
        'train_data_batch_9',
        'train_data_batch_10',
    ]
    valid_list = [
        'val_data',
    ]

    def __init__(self, root,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 use_num_of_class_only=DATA_DESC['num_classes']):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.transform = transform
        self.train = train

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.valid_list
        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for i, file_name in enumerate(downloaded_list):
            file_path = os.path.join(self.root, '{}.npz'.format(file_name))
            entry = np.load(file_path)
            self.data.append(entry["data"])
            self.targets.extend(entry["labels"])
        self.data = np.vstack(self.data).reshape(-1, 3, 64, 64)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        num_classes = 200
        if num_classes < 1000:
            selected_indices = [i for i, label in enumerate(self.targets) if label <= num_classes]
            self.data = self.data[selected_indices]
            self.targets = np.array([self.targets[i] - 1 for i in selected_indices])

    def __repr__(self):
        return "{name}({num} images, {classes} classes)".format(
            name=self.__class__.__name__,
            num=len(self.data),
            classes=len(set(self.targets)),
        )

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class SemiSupervisedImagenet64(SemiSupervisedDataset):

    def load_base_dataset(self, train=False, **kwargs):
        assert self.base_dataset == 'imagenet64', 'Only semi-supervised imagenet64 is supported. Please use correct dataset!'
        self.dataset = ImageNet64(train=train, **kwargs)
        self.dataset_size = len(self.dataset)
        self.num_classes = DATA_DESC['num_classes']
        self.mean_std = (DATA_DESC['mean'], DATA_DESC['std'])


def load_imagenet64(data_dir, logger, use_augmentation='none', use_consistency=False,
                       take_amount=10000, aux_take_amount=None, take_amount_seed=1,
                       add_aux_labels=False, validation=False, pseudo_label_model=None,
                       aux_data_filename=None
                       ):

    test_transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
    if use_augmentation == 'base':
        train_transform = transforms.Compose(
            [transforms.Resize(64), transforms.RandomCrop(64, padding=4), transforms.RandomHorizontalFlip(0.5),
             transforms.ToTensor()])
    else:
        train_transform = test_transform

    train_dataset = SemiSupervisedImagenet64(base_dataset='imagenet64', root=data_dir, train=True,
                                               transform=train_transform,
                                               take_amount=take_amount,
                                               take_amount_seed=take_amount_seed,
                                               add_aux_labels=add_aux_labels,
                                               aux_take_amount=aux_take_amount,
                                               validation=validation,
                                               pseudo_label_model=pseudo_label_model,
                                               aux_data_filename=aux_data_filename,
                                               logger=logger
                                               )
    test_dataset = SemiSupervisedImagenet64(base_dataset='imagenet64', root=data_dir, train=False,
                                                transform=test_transform, logger=logger
                                              )
    if validation:
        val_dataset = ImageNet64(root=data_dir, train=True, transform=test_transform)
        val_dataset = torch.utils.data.Subset(val_dataset, train_dataset.val_indices)
        return train_dataset, test_dataset, val_dataset
    return train_dataset, test_dataset, None

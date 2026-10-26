import os
import pickle
import torchvision.transforms as T
import numpy as np
import torch.utils.data

VALIDATION_FRACTION = 0.2

class SemiSupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, logger, base_dataset='cifar10', take_amount=4000, take_amount_seed=1,
                 add_aux_labels=False, aux_take_amount=None, train=True, validation=False,
                 pseudo_label_model=None, aux_data_filename=None, **kwargs):

        self.base_dataset = base_dataset
        self.load_base_dataset(train, **kwargs)
        self.train = train
        self.logger = logger
        self.origen_len = len(self)
        self.val_indices = []

        if self.train:
            self._prepare_train_split(take_amount, take_amount_seed, validation, **kwargs)
            if pseudo_label_model:
                self._apply_pseudo_labels(self.unsup_indices, pseudo_label_model)
            if aux_data_filename:
                self._load_aux_data(aux_data_filename, aux_take_amount, pseudo_label_model, add_aux_labels, take_amount_seed)
            logger.log(
                f"Train Dataset: label:{len(self.sup_indices)} unlabel:{len(self.unsup_indices)} validation:{len(self.val_indices)}")

        else:
            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []
            logger.log(
                f"Test Dataset: label:{len(self.sup_indices)} unlabel:{len(self.unsup_indices)} validation:{len(self.val_indices)}")

        self.dataset.labels = self.targets

    def _prepare_train_split(self, take_amount, seed, validation, **kwargs):
        _path = os.path.join(kwargs['root'], f'split-{take_amount}-seed-{seed}.npz')
        self.num_classes = len(set(self.dataset.targets))
        total_size = len(self)
        per_class = take_amount // self.num_classes if take_amount else total_size // self.num_classes

        if os.path.exists(_path):
            split_info = np.load(_path)
            unsup_indices = split_info['unsup_indices']
            sup_indices = split_info['sup_indices']

        else:
            rng = np.random.default_rng(seed)
            indices = np.arange(total_size)

            sup_indices, unsup_indices = [], []

            for label in range(self.num_classes):
                label_idx = indices[np.array(self.dataset.targets) == label]
                rng.shuffle(label_idx)
                sup_indices.extend(label_idx[:per_class])
                unsup_indices.extend(label_idx[per_class:])

            np.savez(_path, unsup_indices=unsup_indices, sup_indices=sup_indices)

        for idx in unsup_indices:
            self.targets[idx] = -1

        self.sup_indices = sup_indices
        self.unsup_indices = unsup_indices

        if validation:
            take_val = min(int(take_amount * VALIDATION_FRACTION), 2000)
            val_indices, sup_train = [], []
            for label in range(self.num_classes):
                start = label * per_class
                end = (label + 1) * per_class
                val_indices.extend(sup_indices[start:start + take_val // self.num_classes])
                sup_train.extend(sup_indices[start + take_val // self.num_classes:end])
            self.val_indices = val_indices
            self.sup_indices = sup_train

    def _apply_pseudo_labels(self, indices, model):
        model.eval()
        batch_size = 1024
        acc, total = 0, 0
        transform = T.Compose([T.ToTensor(), T.Resize(32)]) if 'tiny-imagenet' in self.base_dataset else T.ToTensor()

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_data = torch.stack([transform(self.data[k]) if isinstance(self.data[k], np.ndarray)
                                      else self.data[k] for k in batch_idx]).to('cuda')

            with torch.no_grad():
                _, labels = torch.max(model(batch_data), 1)

            for idx, label in zip(batch_idx, labels):
                total += 1
                if self.targets[idx] == int(label):
                    acc += 1
                self.targets[idx] = int(label)
        self.logger.log(f'Pseudo total:{total} acc:{acc * 100 / total:.2f}%')

    def _load_aux_data(self, path, take_amount, pseudo_model, add_aux_labels, seed):
        if os.path.splitext(path)[1] == '.pickle':
            with open(path, 'rb') as f:
                aux = pickle.load(f)
            aux_data = aux['data']
            aux_targets = aux['extrapolated_targets']
        else:
            aux = np.load(path)
            aux_data = aux['image']
            aux_targets = aux['label']

        if take_amount:
            rng_state = np.random.get_state()
            np.random.seed(seed)
            take_inds = np.random.choice(len(aux_data), take_amount, replace=False)
            np.random.set_state(rng_state)
            aux_data = aux_data[take_inds]
            aux_targets = aux_targets[take_inds]

        # tiny-imagenet transform
        if 'tiny-imagenet' in self.base_dataset:
            img_list, label_list = [], []
            for j in range(10):
                tempx = aux_data[aux_targets == self.select[j]]
                tempy = np.full(len(tempx), j)
                img_list.append(tempx)
                label_list.append(tempy)
            aux_data = np.concatenate(img_list, axis=0)
            aux_targets = np.concatenate(label_list, axis=0)


        orig_len = len(self.data)
        self.data = np.concatenate((self.data, aux_data), axis=0)
        self.unsup_indices.extend(range(orig_len, orig_len + len(aux_data)))

        if add_aux_labels:
            self.targets.extend(aux_targets)
        elif pseudo_model:
            self._apply_pseudo_labels(range(orig_len, orig_len + len(aux_data)), pseudo_model)
        else:
            self.targets.extend([-1] * len(aux_data))

    def load_base_dataset(self, **kwargs):
        raise NotImplementedError()

    @property
    def data(self):
        return self.dataset.data

    @data.setter
    def data(self, value):
        self.dataset.data = value

    @property
    def targets(self):
        try:
            return self.dataset.targets
        except AttributeError:
            return self.dataset.labels

    @targets.setter
    def targets(self, value):
        try:
            self.dataset.labels = value
        except AttributeError:
            self.dataset.targets = value

    def __len__(self):
        return len(self.dataset)

    def semi_info(self):
        if self.train:
            return len(self.sup_indices), len(self.unsup_indices), len(self.val_indices)
        return len(self.sup_indices)

    def __getitem__(self, item):
        self.dataset.labels = self.targets
        return self.dataset[item]

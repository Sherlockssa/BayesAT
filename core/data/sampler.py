import numpy as np
import torch


def get_semisup_dataloaders(train_dataset, test_dataset, val_dataset=None, num_batches=None,
                            batch_size=256, batch_size_test=256, num_workers=4, unsup_fraction=0.5, logger=None):
    """
    Return dataloaders with custom sampling of pseudo-labeled data.
    """
    if num_batches == None:
        dataset_size = len(train_dataset)
        num_batches = int(np.ceil(dataset_size / batch_size))
    train_batch_sampler = SemiSupervisedSampler(train_dataset.sup_indices, train_dataset.unsup_indices, batch_size,
                                                unsup_fraction, num_batches=num_batches)
    logger.log(f"Number of batches: {train_batch_sampler.__len__()}")
    batch_size, sup_batch_size = train_batch_sampler.get_batchsize_info()
    logger.log(f"Batch size: sup-{sup_batch_size} unsup-{batch_size - sup_batch_size}.")

    kwargs = {'num_workers': num_workers, 'pin_memory': torch.cuda.is_available()}
    logger.log(kwargs)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, **kwargs)

    if val_dataset:
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_test, shuffle=False, **kwargs)
        return train_dataloader, test_dataloader, val_dataloader
    return train_dataloader, test_dataloader


class SemiSupervisedSampler(torch.utils.data.Sampler):
    """
    Sampling from the labeled and unlabeled data.
    """

    def __init__(self, sup_inds, unsup_inds, batch_size, unsup_fraction=0.5, num_batches=None):
        if unsup_fraction is None or unsup_fraction <= 0 or len(unsup_inds) == 0:
            # self.sup_inds = sup_inds + unsup_inds
            self.sup_inds = sup_inds
            unsup_fraction = 0.0
        else:
            self.sup_inds = sup_inds
            self.unsup_inds = unsup_inds

        self.batch_size = batch_size
        unsup_batch_size = int(batch_size * unsup_fraction)
        self.sup_batch_size = batch_size - unsup_batch_size
        # print(f"Batch size: l-{self.sup_batch_size} ul-{self.batch_size - self.sup_batch_size}.")

        if num_batches is not None:
            self.num_batches = num_batches
        else:
            self.num_batches = int(np.ceil(len(self.sup_inds) / self.sup_batch_size))
        super().__init__(None)

    def __iter__(self):
        batch_counter = 0
        while batch_counter < self.num_batches:
            sup_inds_shuffled = [self.sup_inds[i]
                                 for i in torch.randperm(len(self.sup_inds))]
            for sup_k in range(0, len(self.sup_inds), self.sup_batch_size):
                if batch_counter == self.num_batches:
                    break
                # labeled
                batch = sup_inds_shuffled[sup_k:(sup_k + self.sup_batch_size)]

                if len(batch) < self.sup_batch_size:
                    batch.extend([sup_inds_shuffled[i] for i in torch.randint(high=sup_k,
                                                                              size=(self.sup_batch_size - len(batch),),
                                                                              dtype=torch.int64)])

                if self.sup_batch_size < self.batch_size:
                    # unlabeled
                    batch.extend([self.unsup_inds[i] for i in torch.randint(high=len(self.unsup_inds),
                                                                            size=(self.batch_size - len(batch),),
                                                                            dtype=torch.int64)])

                np.random.shuffle(batch)
                yield batch
                batch_counter += 1

    def __len__(self):
        return self.num_batches

    def get_batchsize_info(self):
        return self.batch_size, self.sup_batch_size
import _pickle as pickle
import argparse
import datetime

import numpy as np
import torch

class SmoothCrossEntropyLoss(torch.nn.Module):
    """
    Cross entropy loss with label smoothing.
    """
    def __init__(self, smoothing=0.0, reduction='mean'):
        super(SmoothCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.reduction = reduction

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CosineLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing LR schedule (used in Carmon et al, 2019).
    """

    def __init__(self, optimizer, max_lr, epochs, last_epoch=-1):
        self.max_lr = max_lr
        self.epochs = epochs
        self._reset()
        super(CosineLR, self).__init__(optimizer, last_epoch)

    def _reset(self):
        self.current_lr = self.max_lr
        self.current_epoch = 1

    def step(self):
        self.current_lr = self.max_lr * 0.5 * (1 + np.cos((self.current_epoch - 1) / self.epochs * np.pi))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
        self.current_epoch += 1
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_lr(self):
        return self.current_lr


def track_bn_stats(model, track_stats=True):
    """
    If track_stats=False, do not update BN running mean and variance and vice versa.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = track_stats


def set_bn_momentum(model, momentum=1):
    """
    Set the value of momentum for all BN layers.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momentum


def str2bool(v):
    """
    Parse boolean using argument parser.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2float(x):
    """
    Parse float and fractions using argument parser.
    """
    if '/' in str(x):
        n, d = x.split('/')
        return float(n)/float(d)
    else:
        try:
            return float(x)
        except:
            raise argparse.ArgumentTypeError('Fraction or float value expected.')


def format_time(elapsed):
    """
    Format time for displaying.
    Arguments:
        elapsed: time interval in seconds.
    """
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def seed(seed=1):
    """
    Seed for PyTorch reproducibility.
    Arguments:
        seed (int): Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
def unpickle_data(filename, mode='rb'):
    """
    Read data from pickled file.
    Arguments:
        filename (str): path to the pickled file.
        mode (str): read mode.
    """
    with open(filename, mode) as pkfile:
        data = pickle.load(pkfile)
    return data


def pickle_data(data, filename, mode='wb'):
    """
    Write data to pickled file.
    Arguments:
        data (Any): data to be written.
        filename (str): path to the pickled file.
        mode (str): write mode.
    """
    with open(filename, mode) as pkfile:
         pickle.dump(data, pkfile)


class NumpyToTensor(object):
    """
    Transforms a numpy.ndarray to torch.Tensor.
    """
    def __call__(self, sample): 
        return torch.from_numpy(sample)


@torch.no_grad()
def dequeue_and_enqueue(keys, queue, queue_ptr, queue_size):
    """
    keys: [B, dim]  当前 batch 特征（已归一化）
    queue: [K, dim]  memory-bank
    queue_ptr: 长为 1 的 tensor
    queue_size: K
    """
    batch_size = keys.shape[0]
    ptr = int(queue_ptr)

    # 如果 batch 过大，直接截断
    if ptr + batch_size > queue_size:
        batch_size = queue_size - ptr

    # 入队
    queue[ptr:ptr + batch_size] = keys[:batch_size]
    ptr = (ptr + batch_size) % queue_size
    queue_ptr[0] = ptr

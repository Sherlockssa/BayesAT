import torch
from torch import nn

eps = 1e-7
class MILoss(nn.Module):
    def __init__(self, opt):
        super(MILoss, self).__init__()
        self.embed1 = Embed(opt.feat_dim, opt.out_feat_dim)
        self.embed2 = Embed(opt.feat_dim, opt.out_feat_dim)
        self.method = opt.contrast_method
        self.criterion = SupConLoss(method=self.method, temperature=opt.contrast_temp, contrast_mode=opt.contrast_mode)

    def forward(self, z1, z2, y=None, mask=None):
        f1 = self.embed1(z1)
        f2 = self.embed2(z2)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)],dim=1)
        if self.method == 'SimCon':
            loss = self.criterion(features, y=None, mask=None)
        elif self.method == 'SupCon':
            loss = self.criterion(features, y, mask)
        else:
            raise NotImplementedError
        return loss



class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, method, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.method = method
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, dim_fea].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device
        batch_size = features.shape[0]
        num_views = features.shape[1]

        if mask is None:
            if labels is None:
                # assert self.method == 'SimCon'
                mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            else:
                # assert self.method == 'SupCon'
                labels = labels.contiguous().view(-1, 1)
                if labels.shape[0] != batch_size:
                    raise ValueError('Num of labels does not match num of features')
                mask = torch.eq(labels, labels.T).float().to(device)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(batch_size, num_views, -1)

        # contrast_feature: the features of all samples in every view
        # torch.unbind(features, dim=1): return a [bsz, dim_fea] metrix for every view
        # torch.cat(-, dim=0): [bsz * n_views, dim_fea]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            # set the first view as anchor
            # anchor_feature: [bsz, dim_fea]
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            # anchor_feature: [bsz * n_views, dim_fea]
            anchor_feature = contrast_feature
            anchor_count = num_views
        else:
            raise ValueError('Unknown contrast_mode: {}'.format(self.contrast_mode))

        # compute logits
        # anchor_dot_contrast: similarity between anchors and features (all samples in every view)
        # if contrast_mode == 'one': [bsz, bsz * n_views]
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        # logits_max: best similarity of every anchor
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, num_views)
        # mask-out self-contrast cases
        # print(mask.shape)
        # print(torch.arange(batch_size * anchor_count).view(-1, 1).shape)
        logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
                0
            )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        # loss = torch.mean(loss.view(anchor_count, batch_size), dim=0)

        return loss

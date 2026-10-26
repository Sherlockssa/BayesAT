import torch
import torch.nn as nn
from torchattacks import FGSM, PGD, AutoAttack
from torchattacks.attack import Attack


def accuracy(true, preds):
    """
    Computes multi-class accuracy.
    Arguments:
        true (torch.Tensor): true labels.
        preds (torch.Tensor): predicted labels.
    Returns:
        Multi-class accuracy.
    """
    accuracy = (torch.softmax(preds, dim=1).argmax(dim=1) == true).sum().float()/float(true.size(0))
    return accuracy.item()


class cw_loss(nn.Module):
    """
    cw loss (Marging loss).
    """
    def __init__(self, confidence=0):
        super().__init__()
        self.confidence = confidence

    def forward(self, outputs, targets, reduction='mean'):
        targets = targets.long()
        one_hot = torch.zeros_like(outputs).scatter_(1, targets.unsqueeze(1), 1)
        real = (outputs * one_hot).sum(1)
        other = ((1 - one_hot) * outputs - one_hot * 1e4).max(1)[0]
        loss = - torch.clamp(real - other + self.confidence, min=0)
        loss = loss.sum()
        return loss


class Natural(Attack):

    def __init__(self, model, **kwargs):
        super().__init__("Natural", model)

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        return images.to(self.device)


class CW_PGD(Attack):

    def __init__(self, model, eps=8/255,
                 alpha=2/255, steps=10, random_start=True):
        super().__init__("CW_PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = cw_loss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


_METRICS = {
    'nat': Natural,
    'fgsm': FGSM,
    'pgd': PGD,
    'cw': CW_PGD,
    'aa': AutoAttack,
}

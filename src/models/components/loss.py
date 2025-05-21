import torch
import torch.nn as nn
import torch.nn.functional as F


def mse_loss(pred, target):
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per predicted patch embedding
    return loss


def norm_mse_loss(pred, target, reduce=True):
    target = F.normalize(target, dim=-1, p=2)
    pred = F.normalize(pred, dim=-1, p=2)
    loss = target * pred
    loss = 2 - 2 * loss.sum(dim=-1)
    return loss.mean() if reduce else loss


def smooth_l1_loss(pred, target):
    return F.smooth_l1_loss(pred, target)


class Loss(nn.Module):
    def __init__(self, loss_type: str, norm_pix_loss: bool | str = True):
        super(Loss, self).__init__()
        self.loss_type = loss_type
        self.norm_pix_loss = norm_pix_loss

        if self.loss_type == 'mse':
            self.loss_fn = mse_loss
        elif self.loss_type == 'norm_mse':
            self.loss_fn = norm_mse_loss
        elif self.loss_type == 'smooth_l1':
            self.loss_fn = smooth_l1_loss
        else:
            assert self.loss_type in ['WE NEED A KNOWN LOSS FN']

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.norm_pix_loss == "center":
            mean = target.mean(dim=-1, keepdim=True)
            target = target - mean
        
        elif self.norm_pix_loss:  # replace by F.layer_norm?
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = self.loss_fn(pred, target)
        return loss

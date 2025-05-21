import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup phase followed by cosine decay.
    
    Args:
        optimizer: The optimizer to modify
        warmup_steps: Number of warmup steps
        start_lr: Initial learning rate during warmup
        ref_lr: Target learning rate after warmup / starting lr for cosine decay
        T_max: Total number of steps (including warmup)
        final_lr: Final learning rate after cosine decay
        last_epoch: The index of last step (-1 default)
    """
    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        final_lr=0.,
        last_epoch=-1
    ):
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.T_max = T_max - warmup_steps
        
        # Initialize with start_lr
        for group in optimizer.param_groups:
            group['initial_lr'] = ref_lr
        
        super().__init__(optimizer, last_epoch)
        
        # Reset learning rate to start_lr
        for group in optimizer.param_groups:
            group['lr'] = start_lr

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            progress = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [self.start_lr + progress * (self.ref_lr - self.start_lr)] * len(self.optimizer.param_groups)
        else:
            # Cosine decay
            progress = float(self.last_epoch - self.warmup_steps) / float(max(1, self.T_max))
            return [max(self.final_lr,
                    self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * 
                    (1. + math.cos(math.pi * progress)))] * len(self.optimizer.param_groups)
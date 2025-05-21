import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineWDScheduler(_LRScheduler):
    """
    Cosine weight decay scheduler that follows PyTorch's scheduler interface.
    
    Args:
        optimizer: The optimizer to modify
        ref_wd: Initial/reference weight decay value
        T_max: Total number of steps
        final_wd: Final weight decay value
        last_epoch: The index of last step (-1 default)
    """
    def __init__(
        self,
        optimizer,
        ref_wd,
        T_max,
        final_wd=0.,
        last_epoch=-1
    ):
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        
        # Store initial weight decay values
        self.base_wds = []
        for group in optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                self.base_wds.append(ref_wd)
            else:
                self.base_wds.append(0.0)
        
        super().__init__(optimizer, last_epoch)

    def get_wd(self):
        """Compute weight decay values for current step"""
        progress = float(self.last_epoch) / float(max(1, self.T_max))
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))

        # Clamp weight decay based on whether we're increasing or decreasing
        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)
            
        return [new_wd if not group.get('WD_exclude', False) else 0.0 
                for group in self.optimizer.param_groups]

    def step(self):
        """Update weight decay values and increment step counter"""
        # Update weight decay values
        wds = self.get_wd()
        for group, wd in zip(self.optimizer.param_groups, wds):
            group['weight_decay'] = wd
            
        # Required by _LRScheduler but we don't use it
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        
        # Increment step counter
        self.last_epoch += 1
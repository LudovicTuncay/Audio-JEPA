from src.optimizers.cosine_wd import CosineWDSchedule
from src.optimizers.warmup_cosine import WarmupCosineSchedule

class CombinedScheduler:
    def __init__(self, lr_scheduler, wd_scheduler):
        self.lr_scheduler = lr_scheduler
        self.wd_scheduler = wd_scheduler
        # Add required attributes for Lightning compatibility
        self.optimizer = lr_scheduler.optimizer  # Use the optimizer from lr_scheduler
        self.base_lrs = lr_scheduler.base_lrs if hasattr(lr_scheduler, 'base_lrs') else None

    def step(self):
        lr = self.lr_scheduler.step()
        wd = self.wd_scheduler.step()
        return lr, wd

    def state_dict(self):
        """Return state dict for checkpointing."""
        return {
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'wd_scheduler': self.wd_scheduler.state_dict()
        }

    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        self.wd_scheduler.load_state_dict(state_dict['wd_scheduler'])
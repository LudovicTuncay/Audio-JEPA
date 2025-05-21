import lightning as L
from src.data.components.audioset_dataset import AudioSetDataset

class AudioSetEpochCallback(L.Callback):
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Update epoch counter in dataset at the start of each epoch."""
        if isinstance(trainer.datamodule.data_train, AudioSetDataset):
            trainer.datamodule.data_train.set_epoch(trainer.current_epoch)
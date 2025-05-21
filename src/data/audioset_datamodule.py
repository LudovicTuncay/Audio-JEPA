from typing import Optional
import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
# from src.data.components.audioset_dataset import collate_audioset_batch


class AudioSetDataModule(L.LightningDataModule):
    """AudioSet DataModule implementing stochastic time-frequency masking strategy.
    
    Implements curriculum-based masking patterns for hierarchical audio representation
    learning following HuBERT/AST paradigms. Supports both global context masks for 
    contrastive learning and local prediction masks for auxiliary tasks.
    """
    
    def __init__(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        batch_size: int,
        num_workers: int,
        val_prop: float = 0.1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        sr: int = 32000,
        # n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 320,
        clip_length: int = 10,
        classes_num: int = 527,
        mask_collator = None,
        transforms = None,
        seed: Optional[int] = None
    ):
        """Initialize AudioSet DataModule.
        
        Args:
            train_file: Path to training HDF5
            eval_file: Path to evaluation HDF5
            val_prop: Proportion of validation set (if set to 0, use eval_file for validation)
            batch_size: Global batch size
            num_workers: DataLoader workers
            pin_memory: Enable memory pinning
            persistent_workers: Maintain worker processes between epochs
            sr: Sample rate (Hz)
            n_mels: MEL filterbank resolution
            n_fft: STFT window size
            hop_length: STFT stride
            clip_length: Temporal context (seconds)
            classes_num: AudioSet taxonomy size
            num_prediction_masks: Number of local prediction tasks
            mask_params: T-F masking configuration
            seed: Random seed for reproducibility
        """
        
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["mask_collator", "train_dataset", "eval_dataset"])
        
        # Data paths
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.val_prop = val_prop
        
        # DataLoader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        
        # Spectrogram parameters
        self.sr = sr
        # self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.clip_length = clip_length
        
        # Task parameters
        self.classes_num = classes_num
        
        self.mask_collator = mask_collator
        self.transforms = transforms
        
        self.seed = seed
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize train/val datasets."""
        
        if self.val_prop > 0:
            train_size = int((1 - self.val_prop) * len(self.train_dataset))
            val_size = len(self.train_dataset) - train_size
            self.data_train, self.data_val = random_split(
                self.train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(self.seed)
            )

        else:
            self.data_train = self.train_dataset
            self.data_val = self.eval_dataset
        
        self.data_test = self.eval_dataset
    
    def train_dataloader(self) -> DataLoader:
        """Build training dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            collate_fn=self.mask_collator,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Build validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            collate_fn=self.mask_collator,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Build test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            collate_fn=self.mask_collator,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )


if __name__ == "__main__":
    _ = AudioSetDataModule()
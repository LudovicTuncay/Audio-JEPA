# src/data/components/audioset_dataset.py

from typing import Dict, Optional, Tuple, List, Union
import torch
import h5py
import io
import numpy as np
from torch.utils.data import Dataset
import torchaudio.transforms as T
import torchaudio
from dataclasses import dataclass
import pandas as pd

from src.utils import audio_utils


@dataclass
class AudioSetBatch:
    """Batch container for self-supervised AudioSet learning with masking.
    """
    waveforms: torch.Tensor
    spectrograms: torch.Tensor
    context_masks: torch.Tensor
    prediction_masks: List[Tuple[torch.Tensor]]
    targets: torch.Tensor
    audio_names: List[str]
    
    # add a method to display the batch
    def __str__(self):
        str = f"Audioset batch:\n"
        str += f" - Batch size:                                   {self.waveforms.shape[0]}\n"
        str += f" - Waveforms [B, samples]:                       {self.waveforms.shape}\n"
        str += f" - Spectrograms [B, C, T, n_mels]:               {self.spectrograms.shape}\n"
        str += f" - Context masks n_masks * [B, n_patches]:       {len(self.context_masks)} * {self.context_masks[0].shape}\n"
        str += f" - Prediction masks n_masks * [B, n_patches]:    {len(self.prediction_masks)} * {self.prediction_masks[0].shape}\n"
        # str += f" - Context masks: [B, n_patches]:      {self.context_masks.shape}\n"
        # str += f" - Target masks: [B, ?]:               {self.prediction_masks.shape}\n"
        # str += f" - N prediction masks:                 {len(self.prediction_masks[0])}\n"
        str += f" - Targets [B, n_classes]:                       {self.targets.shape}\n"
        str += f" - Audio names:                                  {len(self.audio_names)} names\n"
        return str

class AudioSetDataset(Dataset):
    def __init__(
        self,
        hdf5_file: str,
        sr: int = 32000,
        clip_length: int = 10,
        classes_num: int = 527,
        in_mem: bool = False,
        mp3_dataset: bool = True,
        transforms: Optional[List[torch.nn.Module]] = None,
        exclude_csv_path: Optional[str] = None
    ):
        super().__init__()
        
        self.sr = sr
        self.clip_length = clip_length * sr
        self.classes_num = classes_num
        self.mp3_dataset = mp3_dataset
        self.transforms = transforms

        # Handle in-memory HDF5
        if in_mem:
            with open(hdf5_file, 'rb') as f:
                self.hdf5_file = io.BytesIO(f.read())
        else:
            self.hdf5_file = hdf5_file  # Keep as path for lazy loading

        # Get valid indices first (critical for DDP compatibility with excluded indices)
        with h5py.File(self.hdf5_file, 'r') as f:
            total_samples = len(f['audio_name'])
            all_indices = list(range(total_samples))

        # Filter excluded indices
        if exclude_csv_path:
            exclude_df = pd.read_csv(exclude_csv_path)
            exclude_set = set(exclude_df['Index'].values)
            self.valid_indices = [i for i in all_indices if i not in exclude_set]
        else:
            self.valid_indices = all_indices

        self.length = len(self.valid_indices)
        self.dataset_file = None  # Will be initialized lazily

        if self.sr != 32000:
            self.resample = T.Resample(32000, self.sr)

    def open_hdf5(self):
        """Lazy initialization of HDF5 file handle"""
        if isinstance(self.hdf5_file, io.BytesIO):
            self.hdf5_file.seek(0)  # Reset buffer position
        self.dataset_file = h5py.File(self.hdf5_file, 'r')

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict:
        if self.dataset_file is None:
            self.open_hdf5()

        original_idx = self.valid_indices[idx]
        audio_name = self.dataset_file['audio_name'][original_idx].decode()

        # Load core data
        waveform = self._load_waveform(original_idx)
        target = self._load_target(original_idx)

        item = {
            'waveform': waveform,
            'target': target,
            'audio_name': audio_name
        }

        # Apply transforms if specified
        if self.transforms:
            item['transformed_waveform'] = self._apply_transforms(waveform)

        return item

    def _load_waveform(self, idx: int) -> torch.Tensor:
        """Load and preprocess audio waveform"""
        if self.mp3_dataset:
            arr = audio_utils.decode_mp3(self.dataset_file['mp3'][idx])
        else:
            arr = audio_utils.int16_to_float32(self.dataset_file['waveform'][idx])

        waveform = torch.from_numpy(arr).float()
        waveform = audio_utils.normalize_audio(waveform)
        waveform = self.resample(waveform) if self.sr != 32000 else waveform
        return audio_utils.pad_or_truncate(waveform, self.clip_length)

    def _load_target(self, idx: int) -> torch.Tensor:
        """Load and process target labels"""
        target = self.dataset_file['target'][idx]
        if self.mp3_dataset:
            target = np.unpackbits(target, axis=-1, count=self.classes_num).astype(np.float32)
        return torch.from_numpy(target)

    def _apply_transforms(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply transformation pipeline to waveform"""
        # Add batch dimension for transform compatibility
        transformed = waveform.clone().unsqueeze(0)
        
        for transform in self.transforms:
            transformed = transform(transformed)
         
        return transformed

    def __del__(self):
        if self.dataset_file is not None:
            self.dataset_file.close()


def collate_audioset_batch(
    batch: List[Union[Dict, Tuple]]
) -> AudioSetBatch:
    """Collate function for AudioSet batches.
    
    Handles both dictionary and tuple batch formats
    """
    batch = torch.utils.data.default_collate(batch)
    
    if "transformed_waveform" in batch:
        return AudioSetBatch(
            waveforms=batch['waveform'],
            spectrograms=batch['transformed_waveform'],
            context_masks=None,
            prediction_masks=None,
            targets=batch['target'],
            audio_names=batch['audio_name']
        )
    
    else:
        return AudioSetBatch(
            waveforms=batch['waveform'],
            spectrograms=None,
            context_masks=None,
            prediction_masks=None,
            targets=batch['target'],
            audio_names=batch['audio_name']
        )
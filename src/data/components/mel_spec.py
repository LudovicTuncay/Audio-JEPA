import torchaudio.compliance.kaldi as kaldi
import torch
from torch import nn

class MelSpecTransform(nn.Module):
    def __init__(
        self, 
        sr: int,
        n_mels: int, 
        clip_length: int, 
        target_time_bins: int = 1024,
        # normalized: bool = False,
        frame_length: int = None,  # 25ms at 32kHz
        hop_length: int = None,    # 10ms at 32kHz
        # power: float = 2.0,
        f_min: int = 20,         # Kaldi typically uses 20Hz as f_min
        f_max: int = None,
        log: bool = True,
        ):
        
        super().__init__()

        self.sr = sr
        self.n_mels = n_mels
        self.target_time_bins = target_time_bins
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.clip_length = clip_length
        
        self.f_min = f_min
        if f_max is None:
            self.f_max = sr // 2
            
        self.log = log
        
        if hop_length is None:
            # get the hop length to achieve target_time_bins (in ms)
            self.hop_length = (clip_length*1000) / target_time_bins
        
        if frame_length is None:
            self.frame_length = 2.5 * self.hop_length
        
        
    def forward(self, waveform):
        
        # Compute fbank features
        spec = kaldi.fbank(
            waveform - waveform.mean(),  # Subtract mean to ensure zero mean
            sample_frequency=self.sr,
            frame_length=self.frame_length,      # Kaldi expects duration in miliseconds
            frame_shift=self.hop_length,     # Kaldi expects duration in miliseconds
            num_mel_bins=self.n_mels,
            high_freq=self.f_max,
            low_freq=self.f_min,
            use_log_fbank=self.log,  # If True, returns log-filterbank features
            window_type='hanning',
        )
        
        # spec is of shape (time, n_mels)
        
        if spec.shape[0] < self.target_time_bins:
            # Pad to target_time_bins
            pad = self.target_time_bins - spec.shape[0]
            spec = torch.cat([spec, torch.zeros(pad, self.n_mels)], dim=0)
        elif spec.shape[0] > self.target_time_bins:
            # Truncate to target_time_bins
            spec = spec[:self.target_time_bins]

        return spec.unsqueeze(0)
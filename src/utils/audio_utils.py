import numpy as np
import torch
import av
import io
from typing import Union
import torch.nn.functional as F
import torchaudio.transforms as T


def normalize_audio(waveform):
    """
    Normalize audio waveform to be between -1 and 1.
    """
    waveform = waveform - torch.mean(waveform)
    return waveform / torch.max(torch.abs(waveform))+1.e-6


def shift_waveform(waveform, shift):
    """
    Shifts the waveform by a specified number of samples.
    
    Parameters:
    waveform (numpy array): The input audio signal.
    shift (int): The number of samples to shift. Positive values shift right, negative values shift left.
    
    Returns:
    numpy array: The shifted waveform.
    """
    return np.roll(waveform, shift)


def pad_or_truncate(x: Union[np.ndarray, torch.Tensor], audio_length: int) -> Union[np.ndarray, torch.Tensor]:
    """
    Pad or truncate the audio waveform to a fixed length.

    Args:
        x (torch.Tensor): The audio waveform tensor.
        audio_length (int): The desired length of the audio.

    Returns:
        torch.Tensor: The padded or truncated waveform.
        
    Examples:
        >>> x = torch.tensor([1, 2, 3, 4, 5])
        >>> audio_length = 7
        >>> pad_or_truncate(x, audio_length)
        tensor([1, 2, 3, 4, 5, 0, 0])
        
        >>> x = torch.tensor([1, 2, 3, 4, 5])
        >>> audio_length = 3
        >>> pad_or_truncate(x, audio_length)
        tensor([1, 2, 3])
        
        >>> x = torch.tensor([1, 2, 3, 4, 5])
        >>> audio_length = 5
        >>> pad_or_truncate(x, audio_length)
        tensor([1, 2, 3, 4, 5])
    """
    if isinstance(x, torch.Tensor):
        padding = audio_length - x.size(0)
        if padding > 0:
            return torch.nn.functional.pad(x, (0, padding))
        else:
            return x[:audio_length]
    else:
        if len(x) <= audio_length:
            return np.concatenate((x, np.zeros(audio_length - len(x), dtype=np.float32)), axis=0)
        else:
            return x[0: audio_length]
        

def int16_to_float32(x: np.ndarray) -> np.ndarray:
    """
    Converts a numpy array of int16 type to float32 type.

    This function takes an input array of int16 type and performs a conversion to float32 type.
    The conversion is done by dividing each element of the input array by 32767.0 and then
    casting the result to float32 type.

    Args:self
        x (np.ndarray): Input array of int16 type.

    Returns:
        np.ndarray: Converted array of float32 type.
    """
    return (x / 32767.0).astype(np.float32)


def decode_mp3(mp3_arr):
    """
    decodes an array if uint8 representing an mp3 file
    :rtype: np.array
    """
    container = av.open(io.BytesIO(mp3_arr.tobytes()))
    stream = next(s for s in container.streams if s.type == 'audio')
    # print(stream)
    a = []
    for i, packet in enumerate(container.demux(stream)):
        for frame in packet.decode():
            a.append(frame.to_ndarray().reshape(-1))
    waveform = np.concatenate(a)
    if waveform.dtype != 'float32':
        raise RuntimeError("Unexpected wave type")
    return waveform
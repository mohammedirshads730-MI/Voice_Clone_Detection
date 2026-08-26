import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import os

class AudioDeepfakeDataset(Dataset):
    def __init__(self, file_paths, labels, target_sample_rate=16000, max_duration=4.0):
        """
        Args:
            file_paths (list): List of paths to audio files (.wav, .flac, etc.)
            labels (list): 0 for REAL, 1 for FAKE (or vice versa)
            target_sample_rate (int): Required sample rate for Wav2Vec2 (16kHz)
            max_duration (float): Max duration in seconds to truncate/pad audio
        """
        self.file_paths = file_paths
        self.labels = labels
        self.target_sample_rate = target_sample_rate
        self.max_samples = int(target_sample_rate * max_duration)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert stereo to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if sample rate doesn't match target
        if sample_rate != self.target_sample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        # Handle padding or truncation to ensure uniform batch shapes
        waveform = waveform.squeeze(0) # Remove channel dimension
        current_length = waveform.shape[0]

        if current_length > self.max_samples:
            waveform = waveform[:self.max_samples]
        elif current_length < self.max_samples:
            padding = self.max_samples - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform, torch.tensor(label, dtype=torch.long)

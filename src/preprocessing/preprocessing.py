import torchaudio
import torch
import os
from pathlib import Path
import warnings

class PreprocessingService:
    def __init__(self, 
                 sample_rate=16000,
                 n_fft=1024,
                 hop_length=512,
                 n_mels=64,
                 f_min=0,
                 f_max=8000,
                 power=2.0):
        """
        Initialize the preprocessing service with reusable transforms.
        
        Args:
            sample_rate: Target sample rate for processing
            n_fft: FFT window size
            hop_length: Hop length between FFT windows
            n_mels: Number of mel filterbanks
            f_min: Minimum frequency for mel filterbanks
            f_max: Maximum frequency for mel filterbanks
            power: Power for the spectrogram (1 for energy, 2 for power)
        """
        self.sample_rate = sample_rate
        
        # Pre-initialize transforms for efficiency
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=power,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # Store resampler to avoid recreation
        self._resamplers = {}
    
    def compute_melspectrogram(self, file_path=None, target_length=4):
        """
        Compute a mel-scaled spectrogram from an audio waveform.
        
        Args:
            file_path: Path to the audio file
            target_length: Target length of audio in seconds
            
        Returns:
            Mel-spectrogram of shape [channels, n_mels, time]
        """
        if file_path is None:
            raise ValueError("file_path cannot be None")
            
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        waveform, original_sample_rate = self._load_audio_file(file_path)
        waveform = self._preprocess_waveform(
            waveform,
            original_sample_rate,
            target_sample_rate=self.sample_rate,
            target_length=target_length,
        )

        # Use pre-initialized transforms
        melspec = self.mel_spectrogram(waveform)
        
        # Convert to decibels
        melspec = self.amplitude_to_db(melspec)
        
        return melspec

    def _load_audio_file(self, file_path):
        """
        Load an audio file and return the waveform and sample rate.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (waveform, sample_rate)
        """
        try:
            # Validate file extension
            valid_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in valid_extensions:
                warnings.warn(f"Audio file extension {file_ext} may not be supported")
            
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Validate loaded audio
            if waveform.numel() == 0:
                raise ValueError("Loaded audio is empty")
                
            return waveform, sample_rate
            
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            # Return silence instead of zeros for more realistic fallback
            return torch.zeros(1, self.sample_rate), self.sample_rate

    def _preprocess_waveform(self, waveform, sample_rate, target_sample_rate=16000, target_length=4):
        """
        Preprocess the audio waveform.
        
        Args:
            waveform: Audio waveform tensor
            sample_rate: Original sample rate of the audio
            target_sample_rate: Target sample rate for resampling
            target_length: Target length of audio in seconds
            
        Returns:
            Preprocessed waveform tensor
        """
        # Resample if necessary
        waveform = self._resample_audio(waveform, sample_rate, target_sample_rate)
        
        # Convert to mono if stereo
        waveform = self._convert_to_mono(waveform)
        
        # Trim or pad to target length
        target_length_samples = target_length * target_sample_rate
        waveform = self._trim_or_pad_waveform(waveform, target_length_samples)
        
        # Normalize the waveform
        waveform = self._normalize_waveform(waveform)
        
        return waveform
        
    def _resample_audio(self, waveform, original_sample_rate, target_sample_rate=16000):
        """
        Resample the audio waveform to a target sample rate.
        
        Args:
            waveform: Audio waveform tensor
            original_sample_rate: Original sample rate of the audio
            target_sample_rate: Target sample rate
            
        Returns:
            Resampled waveform tensor
        """
        if original_sample_rate != target_sample_rate:
            # Cache resamplers to avoid recreation
            resampler_key = (original_sample_rate, target_sample_rate)
            if resampler_key not in self._resamplers:
                self._resamplers[resampler_key] = torchaudio.transforms.Resample(
                    orig_freq=original_sample_rate,
                    new_freq=target_sample_rate
                )
            
            waveform = self._resamplers[resampler_key](waveform)
        
        return waveform
    
    def _convert_to_mono(self, waveform):
        """
        Convert a stereo waveform to mono by averaging channels.
        
        Args:
            waveform: Audio waveform tensor
            
        Returns:
            Mono waveform tensor
        """
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform
    
    def _trim_or_pad_waveform(self, waveform, target_length=16000, strategy='start'):
        """
        Trim or pad the waveform to a target length.
        
        Args:
            waveform: Audio waveform tensor
            target_length: Target length in samples
            strategy: 'start', 'center', 'random', or 'loudest' - how to select the segment
            
        Returns:
            Waveform tensor trimmed or padded to target length
        """
        current_length = waveform.shape[1]
        
        if current_length > target_length:
            # Choose trimming strategy
            if strategy == 'start':
                start_idx = 0
            elif strategy == 'center':
                start_idx = (current_length - target_length) // 2
            elif strategy == 'random':
                start_idx = torch.randint(0, current_length - target_length + 1, (1,)).item()
            elif strategy == 'loudest':
                # Find the segment with highest energy
                energy = waveform.pow(2).squeeze(0)
                # Use sliding window to find loudest segment
                window_energy = torch.nn.functional.conv1d(
                    energy.unsqueeze(0).unsqueeze(0),
                    torch.ones(1, 1, target_length),
                    padding=0
                ).squeeze()
                start_idx = torch.argmax(window_energy).item()
            else:
                start_idx = 0
                
            waveform = waveform[:, start_idx:start_idx + target_length]
        else:
            # Pad with zeros
            pad_amount = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        
        return waveform
    
    def _normalize_waveform(self, waveform, eps=1e-8):
        """
        Normalize the waveform to have zero mean and unit variance.
        
        Args:
            waveform: Audio waveform tensor
            eps: Small value to prevent division by zero
            
        Returns:
            Normalized waveform tensor
        """
        mean = waveform.mean()
        std = waveform.std()
        
        # Prevent division by zero for silent or constant audio
        if std < eps:
            # For silent audio, return zeros
            # For constant audio, just remove the DC component
            return waveform - mean
        
        waveform = (waveform - mean) / std
        return waveform
    
    def _apply_audio_augmentations(self, waveform, augment=False):
        """
        Apply audio augmentations for training data diversity.
        
        Args:
            waveform: Audio waveform tensor
            augment: Whether to apply augmentations
            
        Returns:
            Augmented waveform tensor
        """
        if not augment:
            return waveform
            
        # Random gain (volume) adjustment
        if torch.rand(1) < 0.3:
            gain = torch.uniform(0.7, 1.3, (1,))
            waveform = waveform * gain
            
        # Add slight noise
        if torch.rand(1) < 0.2:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
            
        return waveform
    
    def get_audio_info(self, file_path):
        """
        Get information about an audio file without loading the full waveform.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary with audio information
        """
        try:
            info = torchaudio.info(file_path)
            return {
                'sample_rate': info.sample_rate,
                'num_frames': info.num_frames,
                'num_channels': info.num_channels,
                'duration_seconds': info.num_frames / info.sample_rate,
                'encoding': info.encoding,
                'bits_per_sample': info.bits_per_sample
            }
        except Exception as e:
            return {'error': str(e)}
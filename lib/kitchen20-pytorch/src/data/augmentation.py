import torch
import torchaudio
import random

class AudioAugmentation:
    """Audio augmentation class implementing various transformation methods."""
    
    def __init__(self, sample_rate=16000, p=0.5):
        """
        Args:
            sample_rate: Audio sample rate
            p: Probability of applying each augmentation
        """
        self.sample_rate = sample_rate
        self.p = p
        
    def time_shift(self, waveform, shift_factor=0.2):
        """
        Shift the audio in time.
        
        Args:
            waveform: Audio tensor of shape [channels, time]
            shift_factor: Maximum shift as a fraction of total length
            
        Returns:
            Time-shifted waveform
        """
        if random.random() < self.p:
            length = waveform.shape[1]
            shift_amount = int(random.random() * shift_factor * length)
            
            if random.random() < 0.5:
                shift_amount = -shift_amount
                
            waveform = torch.roll(waveform, shift_amount, dims=1)
            
        return waveform
    
    def add_noise(self, waveform, noise_level=0.005):
        """
        Add random noise to the waveform.
        
        Args:
            waveform: Audio tensor of shape [channels, time]
            noise_level: Standard deviation of the noise
            
        Returns:
            Waveform with added noise
        """
        if random.random() < self.p:
            noise = torch.randn_like(waveform) * noise_level
            waveform = waveform + noise
            
        return waveform
    
    def change_speed(self, waveform, speed_range=(0.85, 1.15)):
        """
        Change the speed of the audio.
        
        Args:
            waveform: Audio tensor of shape [channels, time]
            speed_range: Range of speed factors
            
        Returns:
            Speed-modified waveform
        """
        if random.random() < self.p:
            speed_factor = random.uniform(speed_range[0], speed_range[1])
            
            # Use torchaudio's speed function
            effects = [
                ["speed", str(speed_factor)],
                ["rate", str(self.sample_rate)]
            ]
            
            # Apply to each channel independently
            channels = []
            for ch in range(waveform.shape[0]):
                aug_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                    waveform[ch].unsqueeze(0),
                    self.sample_rate,
                    effects
                )
                channels.append(aug_waveform)
                
            # Combine channels back
            if len(channels) == 1:
                waveform = channels[0]
            else:
                waveform = torch.cat(channels, dim=0)
                
        return waveform
    
    def pitch_shift(self, waveform, pitch_range=(-3, 3)):
        """
        Shift the pitch of the audio.
        
        Args:
            waveform: Audio tensor of shape [channels, time]
            pitch_range: Range of semitones to shift
            
        Returns:
            Pitch-shifted waveform
        """
        if random.random() < self.p:
            n_steps = random.uniform(pitch_range[0], pitch_range[1])
            
            effects = [
                ["pitch", str(n_steps * 100)],  # Pitch shift in cents
                ["rate", str(self.sample_rate)]
            ]
            
            # Apply to each channel independently
            channels = []
            for ch in range(waveform.shape[0]):
                aug_waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                    waveform[ch].unsqueeze(0),
                    self.sample_rate,
                    effects
                )
                channels.append(aug_waveform)
                
            # Combine channels back
            if len(channels) == 1:
                waveform = channels[0]
            else:
                waveform = torch.cat(channels, dim=0)
                
        return waveform
    
    def __call__(self, waveform):
        """Apply a series of random augmentations to the waveform."""
        waveform = self.time_shift(waveform)
        waveform = self.add_noise(waveform)
        waveform = self.change_speed(waveform)
        waveform = self.pitch_shift(waveform)
        
        return waveform


# Spectrogram augmentations for after feature extraction
class SpectrogramAugmentation:
    """Spectrogram augmentation class implementing various transformation methods."""
    
    def __init__(self, p=0.5):
        """
        Args:
            p: Probability of applying each augmentation
        """
        self.p = p
        
    def time_masking(self, spectrogram, max_time_mask=10):
        """
        Apply time masking to the spectrogram.
        
        Args:
            spectrogram: Spectrogram tensor [channels, freq, time]
            max_time_mask: Maximum time mask length
            
        Returns:
            Time-masked spectrogram
        """
        if random.random() < self.p:
            time_mask = torchaudio.transforms.TimeMasking(max_time_mask)
            spectrogram = time_mask(spectrogram)
            
        return spectrogram
    
    def freq_masking(self, spectrogram, max_freq_mask=10):
        """
        Apply frequency masking to the spectrogram.
        
        Args:
            spectrogram: Spectrogram tensor [channels, freq, time]
            max_freq_mask: Maximum frequency mask length
            
        Returns:
            Frequency-masked spectrogram
        """
        if random.random() < self.p:
            freq_mask = torchaudio.transforms.FrequencyMasking(max_freq_mask)
            spectrogram = freq_mask(spectrogram)
            
        return spectrogram
    
    def __call__(self, spectrogram):
        """Apply a series of random augmentations to the spectrogram."""
        spectrogram = self.time_masking(spectrogram)
        spectrogram = self.freq_masking(spectrogram)
        
        return spectrogram
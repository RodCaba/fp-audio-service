import torchaudio

def compute_melspectrogram(
    waveform,
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=64,
    f_min=0,
    f_max=8000,
    power=2.0,
):
    """
    Compute a mel-scaled spectrogram from an audio waveform.
    
    Args:
        waveform: Audio tensor of shape [channels, time]
        sample_rate: Audio sample rate
        n_fft: FFT window size
        hop_length: Hop length between FFT windows
        n_mels: Number of mel filterbanks
        f_min: Minimum frequency for mel filterbanks
        f_max: Maximum frequency for mel filterbanks
        power: Power for the spectrogram (1 for energy, 2 for power)
        
    Returns:
        Mel-spectrogram of shape [channels, n_mels, time]
    """
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=power,
    )
    
    melspec = mel_spectrogram(waveform)
    
    # Convert to decibels
    melspec = torchaudio.transforms.AmplitudeToDB()(melspec)
    
    return melspec


def compute_mfcc(
    waveform,
    sample_rate=16000,
    n_mfcc=13,
    n_fft=1024,
    hop_length=512,
    n_mels=64,
    f_min=0,
    f_max=8000,
):
    """
    Compute MFCC features from an audio waveform.
    
    Args:
        waveform: Audio tensor of shape [channels, time]
        sample_rate: Audio sample rate
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length between FFT windows
        n_mels: Number of mel filterbanks
        f_min: Minimum frequency for mel filterbanks
        f_max: Maximum frequency for mel filterbanks
        
    Returns:
        MFCC features of shape [channels, n_mfcc, time]
    """
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            'n_fft': n_fft,
            'hop_length': hop_length,
            'n_mels': n_mels,
            'f_min': f_min,
            'f_max': f_max,
        }
    )
    
    return mfcc_transform(waveform)


def extract_features(waveform, sample_rate=16000, feature_type='melspectrogram'):
    """
    Extract audio features from a waveform.
    
    Args:
        waveform: Audio tensor of shape [channels, time]
        sample_rate: Audio sample rate
        feature_type: Type of features to extract ('melspectrogram' or 'mfcc')
        
    Returns:
        Audio features
    """
    if feature_type == 'melspectrogram':
        return compute_melspectrogram(waveform, sample_rate=sample_rate)
    elif feature_type == 'mfcc':
        return compute_mfcc(waveform, sample_rate=sample_rate)
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")


class FeatureExtractor:
    """Audio feature extractor class."""
    
    def __init__(self, 
                 feature_type='melspectrogram',
                 sample_rate=16000,
                 n_fft=1024,
                 hop_length=512,
                 n_mels=64,
                 n_mfcc=13,
                 f_min=0,
                 f_max=8000):
        """
        Args:
            feature_type: Type of features to extract ('melspectrogram' or 'mfcc')
            sample_rate: Target sample rate
            n_fft: FFT window size
            hop_length: Hop length between FFT windows
            n_mels: Number of mel filterbanks
            n_mfcc: Number of MFCC coefficients (if feature_type='mfcc')
            f_min: Minimum frequency
            f_max: Maximum frequency
        """
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.f_min = f_min
        self.f_max = f_max
        
        if feature_type == 'melspectrogram':
            self.transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
            )
            self.db_transform = torchaudio.transforms.AmplitudeToDB()
        elif feature_type == 'mfcc':
            self.transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={
                    'n_fft': n_fft,
                    'hop_length': hop_length,
                    'n_mels': n_mels,
                    'f_min': f_min,
                    'f_max': f_max,
                }
            )
            self.db_transform = None
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
    
    def __call__(self, waveform):
        """Extract features from waveform."""
        features = self.transform(waveform)
        
        if self.db_transform is not None:
            features = self.db_transform(features)
        
        return features
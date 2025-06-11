import torchaudio
import torch


def load_audio(file_path):
    """
    Load an audio file and return the waveform and sample rate.
    
    Args:
        file_path (str): Path to the audio file.
        
    Returns:
        tuple: (waveform, sample_rate)
    """
    try:
      waveform, sample_rate = torchaudio.load(file_path)
      return waveform, sample_rate
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return torch.zeros(1, 16000), 16000  # Return a dummy tensor if loading fails
    
def preprocess_audio(
      waveform,
      original_sample_rate,
      target_sample_rate=16000,
      target_length=4,
):
    """
    Preprocess the audio waveform to have a consistent length and sample rate.

    Args:
        waveform (Tensor): The audio waveform.
        original_sample_rate (int): The original sample rate of the audio.
        target_sample_rate (int, optional): The target sample rate. Defaults to 16000.
        target_length (int, optional): Target number of samples.

    Returns:
        Tensor: The preprocessed audio waveform.
    """
    # Convert to mono if stereo
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample the audio if the sample rate is different
    if original_sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=original_sample_rate,
            new_freq=target_sample_rate
        )(waveform)

    # Adjust length
    current_length = waveform.shape[1]
    # Trim or pad the waveform to the target length
    if current_length > target_length:
        waveform = waveform[:, :target_length]
    else:
        waveform = torch.nn.functional.pad(waveform, (0, target_length - current_length))

    return waveform

def normalize_audio(waveform):
    """
    Normalize the audio waveform to have zero mean and unit variance.

    Args:
        waveform (Tensor): The audio waveform.

    Returns:
        Tensor: The normalized audio waveform.
    """
    if torch.max(torch.abs(waveform)) > 0:
        waveform = waveform / torch.max(torch.abs(waveform))
    return waveform
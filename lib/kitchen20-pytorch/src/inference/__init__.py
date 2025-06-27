import argparse
import torch

class AudioTransform:
  """
  Wrapper class for audio transformations.

  Args:
      feature_extractor (callable): Function to extract features from audio.
      augmentation (callable, optional): Augmentation function to apply to the audio.
      spec_augmentation (callable, optional): Spectrogram augmentation function.
  """
  def __init__(
      self,
      feature_extractor,
      augmentation=None,
      spec_augmentation=None,
  ):
    self.feature_extractor = feature_extractor
    self.augmentation = augmentation
    self.spec_augmentation = spec_augmentation

  def __call__(self, waveform):
    if self.augmentation:
      waveform = self.augmentation(waveform)
    
    features = self.feature_extractor(waveform)

    if self.spec_augmentation:
      features = self.spec_augmentation(features)
    
    return features
  
def main():
    """
    Entry point for model training
    """
    parser = argparse.ArgumentParser(description="Train a Kitchen20 model")
    parser.add_argument('--data_path', type=str, required=True, help='Path to audio data directory')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to metadata CSV file')
    
    args = parser.parse_args()

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create feature extractors
    
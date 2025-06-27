import argparse
import json
import torch
from torch import nn, optim
from pathlib import Path
from common.preprocessing import FeatureExtractor
from ..data.dataset import create_data_loaders
from ..models.cnn_model import CNNModel
from ..training.train import train_model

class AudioTransform:
  """
  Wrapper class for audio transformations.
  """

  def __init__(self, feature_extractor, augmentation=None, spec_augmentation=None):
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
  parser.add_argument('--feature_type', type=str, default='melspectrogram', choices=['melspectrogram', 'mfcc'], help='Type of audio features to extract')

  args = parser.parse_args()

  # Set up device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  # Create feature extractors
  feature_extractor = FeatureExtractor(feature_type=args.feature_type)

  # Create audio transform
  train_transform = AudioTransform(
      feature_extractor=feature_extractor,
      augmentation=None,
      spec_augmentation=None
  )
  val_transform = AudioTransform(
      feature_extractor=feature_extractor,
      augmentation=None,  # Typically no augmentation for validation
      spec_augmentation=None
  )
  print("Feature extractor and audio transform initialized.")

  transforms = {
      'train': train_transform,
      'val': val_transform
  }


  # Load dataset and create data loaders
  train_loader, val_loader, class_names = create_data_loaders(
    csv_path=args.csv_path,
    audio_dir=args.data_path,
    transforms=transforms,
  )

  print(f"Number of classes: {len(class_names)}")
  print(f"Class names: {class_names}")

  # Initialize model
  model = CNNModel(num_classes=len(class_names)).to(device)
  print("Model initialized.")

  # Create checkpoint directory
  checkpoint_dir = Path('checkpoints')
  checkpoint_dir.mkdir(exist_ok=True, parents=True)

  # Save class names to a file
  with open(checkpoint_dir / 'class_names.json', 'w') as f:
    json.dump(class_names, f)
  
  # Set up training
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode='min',
    factor=0.5,
    patience=5,
  )

  # Train the model
  print("Starting training...")
  history, best_model_path = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    scheduler=scheduler,
    checkpoint_dir=checkpoint_dir
  )
  print("Training completed.")
  print(f"Best model saved at: {best_model_path}")

  # Save training history
  with open(checkpoint_dir / 'training_history.json', 'w') as f:
    for k in history:
      history[k] = [float(v) for v in history[k]]
    json.dump(history, f)

if __name__ == "__main__":
  main()
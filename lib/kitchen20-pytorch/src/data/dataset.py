from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from common.audio import load_audio, preprocess_audio

class Kitchen20Dataset(Dataset):
    def __init__(
            self,
            csv_path,
            audio_dir=None,
            fold=None,
            transform=None,
            target_sample_rate=16000,
            target_length=4,
            training=True,
    ):
      """
      Kitchen 20 dataset for audio classification.
      Args:
          csv_path (str): Path to the CSV file containing metadata.
          audio_dir (str, optional): Directory containing audio files. Defaults to None.
          fold (int, optional): Fold number for cross-validation. Defaults to None.
          transform (callable, optional): Transform to apply to the audio data. Defaults to None.
          target_sample_rate (int, optional): Target sample rate for audio files. Defaults to 16000.
          target_length (int, optional): Target length of audio clips in seconds. Defaults to 4.
          training (bool, optional): Whether the dataset is for training or evaluation. Defaults to True.
      """
      self.df = pd.read_csv(csv_path)

      # Filter by fold if specified
      if fold is not None:
         if training:
            # Use all folds except the specified one for training
            self.df = self.df[self.df['fold'] != fold]
         else:
            # Use the specified fold for evaluation
            self.df = self.df[self.df['fold'] == fold]

      self.audio_dir = audio_dir
      self.transform = transform
      self.target_sample_rate = target_sample_rate
      self.target_length = target_length
      self.num_samples = int(target_sample_rate * target_length)

      # Get unique labels
      self.labels = sorted(self.df['category'].unique())
      self.label_to_index = {label: index for index, label in enumerate(self.labels)}

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx):
      row = self.df.iloc[idx]
      # Get audio path
      if self.audio_dir:
         audio_path = os.path.join(self.audio_dir, row['path'])
      else:
        audio_path = row['path']
       
      # Load audio
      waveform, sample_rate = load_audio(audio_path)

      # Process audio to have consistent length and sample rate
      waveform = preprocess_audio(
         waveform,
         original_sample_rate=sample_rate,
         target_sample_rate=self.target_sample_rate,
         target_length=self.num_samples
        )
      
      # Apply transformation if specified
      if self.transform:
         waveform = self.transform(waveform)
      # Get label index
      label_index = self.label_to_index[row['category']]
      return waveform, label_index
    
    def get_labels(self):
       """
       Returns the list of unique labels in the dataset.
       """
       return self.labels

def create_data_loaders(
      csv_path,
      audio_dir=None,
      fold=5,
      batch_size=32,
      num_workers=4,
      transforms=None
):
   """
   Create training and validation data loaders

   Args:
       csv_path (str): Path to the CSV file containing metadata.
       audio_dir (str, optional): Directory containing audio files. Defaults to None.
       fold (int, optional): Fold number for cross-validation. Defaults to 5.
       batch_size (int, optional): Batch size for data loaders. Defaults to 32.
       transforms (callable, optional): Transform to apply to the audio data. Defaults to None.
   Returns:
      train_loader (DataLoader): DataLoader for training data.
      val_loader (DataLoader): DataLoader for validation data.
      class_names (list): List of class names.
   """
   if transforms is None:
      transforms = {'train': None, 'val': None}
   
   # Create datasets
   train_dataset = Kitchen20Dataset(
      csv_path=csv_path,
      audio_dir=audio_dir,
      fold=fold,
      transform=transforms.get('train'),
      training=True
   )
   val_dataset = Kitchen20Dataset(
      csv_path=csv_path,
      audio_dir=audio_dir,
      fold=fold,
      transform=transforms.get('val'),
      training=False
   )
   # Create data loaders
   train_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True
   )

   val_loader = DataLoader(
      val_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True
   )

   return train_loader, val_loader, train_dataset.get_labels()
import torch
import argparse
import onnxruntime as ort
import numpy as np
from pathlib import Path
from common.preprocessing import FeatureExtractor
from common.audio import load_audio, preprocess_audio

class AudioPredictor:
  """
  Audio classification predictor.
  """
  def __init__(
      self,
      model_path: str,
      feature_type='melspectrogram',
      device=None,
  ):
    """
    Args:
        model_path (str): Path to the trained model.
        feature_type (str): Type of audio features to use ('melspectrogram' or 'mfcc').
        device (str): Device to run the model on (None for auto-detection).
    """
    if device is None:
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
      self.device = torch.device(device)
    print(f"Using device: {self.device}")

    # Load the model
    self.feature_type = feature_type
    self.model_path = Path(model_path)
    self.model_format = Path(model_path).suffix
    self.model = self._load_model()

    # Feature extractor
    self.feature_extractor = FeatureExtractor(
      feature_type=feature_type,
    )

    # Kitchen20 class names
    self.class_names = [
        'blender', 'boiling-water', 'book', 'chopping', 'clean-dishes', 
        'cupboard', 'cutlery', 'dishwasher', 'drawer', 'eating', 
        'food-processor', 'fridge', 'kettle', 'microwave', 'mixer',
        'pan', 'plates', 'sink', 'tap', 'trash'
    ]

  def _load_model(self):
    """
    Load the trained model based on the file format.
    
    Returns:
        torch.nn.Module: Loaded model.
    """
    if self.model_format == '.pt':
      # Load TorchScript model
      model = torch.jit.load(self.model_path, map_location=self.device)
      return model
    
    elif self.model_format == '.onnx':
      options = ort.SessionOptions()
      options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

      # Use provider based on device
      if self.device.type == 'cuda':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
      else:
        providers = ['CPUExecutionProvider']
      
      self.onnx_session = ort.InferenceSession(
        self.model_path,
        providers=providers,
        sess_options=options
      )
      return None
    
    else:
      raise ValueError(f"Unsupported model format: {self.model_format}. Use .pt or .onnx.")

  def predict(self, audio_path, target_sr=16000, target_length=4):
    """
    Predict the class of an audio file.
    Args:
        audio_path (str): Path to the audio file.
        target_sr (int): Target sample rate for audio.
        target_length (int): Target length of audio in seconds.

    Returns:
        tuple: (predicted_class_name, confidence, all_probabilities)
    """
    waveform, sample_rate = load_audio(audio_path)
    waveform = preprocess_audio(
      waveform,
      sample_rate,
      target_sample_rate=target_sr,
      target_length=target_length * target_sr,
    )

    # Extract features
    features = self.feature_extractor(waveform)

    # Make prediction based on model format
    if self.model_format == '.pt':
      features = features.unsqueeze(0).to(self.device)  # Add batch dimension
      with torch.no_grad():        
        outputs = self.model(features)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
      # Get predicted class
      predicted_class_idx = torch.argmax(probabilities).item()
      predicted_class_name = self.class_names[predicted_class_idx]
      confidence = probabilities[predicted_class_idx].item()
      all_probabilities = probabilities.cpu().numpy()
    elif self.model_format == '.onnx':
      _, n_mels, time_frames = features.shape
      required_time_frames = 128

      if time_frames != required_time_frames:
        print(f"Resizing features from shape {features.shape} to (1, 1, {n_mels}, {required_time_frames})")
        features = torch.nn.functional.interpolate(
          features.unsqueeze(0),
          size=(n_mels, required_time_frames),
          mode='bilinear',
          align_corners=False
        ).squeeze(0)
      features = features.unsqueeze(0).numpy()  # Convert to numpy array

      # Run inference
      outputs = self.onnx_session.run(
        None,
        {'input': features}
      )[0]
      all_probabilities = self._softmax(outputs[0])
      predicted_class_idx = np.argmax(all_probabilities)
      confidence = all_probabilities[predicted_class_idx]
      predicted_class_name = self.class_names[predicted_class_idx]
    else:
      raise ValueError(f"Unsupported model format: {self.model_format}. Use .pt or .onnx.")

    return predicted_class_name, confidence, all_probabilities
  
  def _softmax(self, x):
    """
    Apply softmax to the output tensor.

    """
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()
  
def main():
  """
  Script entry point for testing the predictor.
  """
  parser = argparse.ArgumentParser(
    description="Predict audio class using a trained model."
  )
  parser.add_argument("audio_path", type=str, help="Path to the audio file.")
  parser.add_argument("--model", type=str, default='checkpoints/best_model.pth', help="Path to the trained model.")
  args = parser.parse_args()

  predictor = AudioPredictor(
    model_path=args.model,
    feature_type='melspectrogram',
  )

  audio_path = Path(args.audio_path)

  predicted_class_name, confidence, all_probabilities = predictor.predict(
    audio_path=audio_path
  )

  print(f"Predicted class: {predicted_class_name}")
  print(f"Confidence: {confidence:.4f}")
  print("Top-3 class probabilities:")
  top_indices = all_probabilities.argsort()[-3:][::-1]
  for idx in top_indices:
    print(f"{predictor.class_names[idx]}: {all_probabilities[idx]:.4f}")

if __name__ == "__main__":
  main()
import argparse
import torch
import os
from ..models.cnn_model import CNNModel

def export_model(model, export_path, format='torchscript'):
  """
  Export the model to the specified format.
  Args:
      model (torch.nn.Module): The model to export.
      export_path (str): Path to save the exported model.
      format (str): Format to export the model ('torchscript', 'onnx').
  """
  model.eval()  # Set the model to evaluation mode

  # Create directory if it doesn't exist
  os.makedirs(os.path.dirname(export_path), exist_ok=True)
  export_path += '.pt' if format == 'torchscript' else '.onnx'
  
  example_input = torch.randn(1, 1, 64, 128)  # Example input shape (Batch, channel, n_mels, time)

  if format == 'torchscript':
    scripted_model = torch.jit.trace(model, example_input)
    torch.jit.save(scripted_model, export_path)
    print(f"Model exported to {export_path} in TorchScript format.")
  elif format == 'onnx':
    torch.onnx.export(
      model,
      example_input,
      export_path,
      export_params=True,
      opset_version=12,
      do_constant_folding=True,
      input_names=['input'],
      output_names=['output'],
      dynamic_axes={
        'input': {0: 'batch_size'},  # Variable batch size
        'output': {0: 'batch_size'}
      }
    )
    print(f"Model exported to {export_path} in ONNX format.")
  else:
    raise ValueError("Unsupported export format. Use 'torchscript' or 'onnx'.")


def main():
    """
    Main function to export the model.
    """
    parser = argparse.ArgumentParser(description="Export trained model to TorchScript or ONNX format.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint file. (e.g., best_model.pth)"
    )
    parser.add_argument(
        "--export_path",
        type=str,
        required=True,
        help="Path to save the exported model file. (e.g., exported_model.pt or exported_model.onnx)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=['torchscript', 'onnx'],
        default='onnx',
        help="Format to export the model. (default: onnx)"
    )
    args = parser.parse_args()
    model = CNNModel()
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode

    export_model(model, args.export_path, format=args.format)

if __name__ == "__main__":
  main()
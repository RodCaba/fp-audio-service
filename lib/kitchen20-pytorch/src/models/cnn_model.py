import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
  """
  Convolutional Neural Network (CNN) model for audio classification.
  """
  def __init__(
      self,
      num_classes=20,
      n_mels=64,
      n_time_frames=128,
      dropout_rate=0.2,
  ):
    """
    Args:
        num_classes (int): Number of output classes.
        n_mels (int): Number of mel frequency bins.
        n_time_frames (int): Number of time frames in the input.
        dropout_rate (float): Dropout rate for regularization.
    """
    super(CNNModel, self).__init__()
    
    self.n_mels = n_mels
    self.n_time_frames = n_time_frames

    # Convolutional layers
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(32)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(64)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.bn3 = nn.BatchNorm2d(128)
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.bn4 = nn.BatchNorm2d(256)
    
    # Use adaptive pooling to handle variable input sizes
    self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

    # Fully connected layers, we know the output size after conv layers will be (256, 4, 4)
    self.fc1 = nn.Linear(256 * 4 * 4, 512)
    self.bn5 = nn.BatchNorm1d(512)
    self.dropout = nn.Dropout(dropout_rate)

    self.fc2 = nn.Linear(512, 128)
    self.bn6 = nn.BatchNorm1d(128)
    self.dropout2 = nn.Dropout(dropout_rate)

    self.fc3 = nn.Linear(128, num_classes)

  def _forward_conv(self, x):
    """
    Forward pass through the convolutional layers.
    """
    x = F.relu(self.bn1(self.conv1(x)))
    x = self.pool1(x)

    x = F.relu(self.bn2(self.conv2(x)))
    x = self.pool2(x)

    x = F.relu(self.bn3(self.conv3(x)))
    x = self.pool3(x)

    x = F.relu(self.bn4(self.conv4(x)))
    # Apply adaptive pooling to ensure consistent output size
    x = self.adaptive_pool(x)

    return x

  def forward(self, x):
    """
    Forward pass through the model.
    """
    x = self._forward_conv(x)

    # Flatten the tensor
    x = x.view(x.size(0), -1)

    # Fully connected layers
    x = F.relu(self.bn5(self.fc1(x)))
    x = self.dropout(x)

    x = F.relu(self.bn6(self.fc2(x)))
    x = self.dropout2(x)

    x = self.fc3(x)
    
    return x
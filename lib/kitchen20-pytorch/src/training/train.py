from tqdm import tqdm
from pathlib import Path
import torch
import time

def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):
  """
  Train the model for one epoch.
  Args:
      model (nn.Module): The model to train.
      dataloader (DataLoader): DataLoader for the training data.
      criterion (nn.Module): Loss function.
      optimizer (Optimizer): Optimizer for updating model parameters.
      device (torch.device): Device to run the training on (CPU or GPU).
  
  Returns:
      tuple: (average loss, accuracy)
  """
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0

  for inputs, targets in tqdm(dataloader, desc="Training"):
    inputs, targets = inputs.to(device), targets.to(device)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update statistics
    running_loss += loss.item() * inputs.size(0)
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
  
  epoch_loss = running_loss / len(dataloader.dataset)
  epoch_acc = correct / total
  print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
  return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on the validation set.
    
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the evaluation on (CPU or GPU).
    
    Returns:
        tuple: (average loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    print(f"Val Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc

def train_model(
      model,
      train_loader,
      val_loader,
      criterion,
      optimizer,
      device,
      num_epochs=50,
      scheduler=None,
      early_stopping_patience=10,
      checkpoint_dir='checkpoints',
):
   """
   Train a model

   Args:
      model (nn.Module): The model to train.
      train_loader (DataLoader): DataLoader for training data.
      val_loader (DataLoader): DataLoader for validation data.
      criterion (nn.Module): Loss function.
      optimizer (Optimizer): Optimizer for updating model parameters.
      device (torch.device): Device to run the training on (CPU or GPU).
      num_epochs (int, optional): Number of epochs to train. Defaults to 50.
      scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Defaults to None.
      early_stopping_patience (int, optional): Patience for early stopping. Defaults to 10.
      checkpoint_dir (str, optional): Directory to save checkpoints. Defaults to 'checkpoints'.
    Returns:
      dict:  Training history containing loss and accuracy for each epoch.
      str: Path to the best model checkpoint.
   """
   # Create checkboint directory if it doesn't exist
   checkpoint_dir = Path(checkpoint_dir)
   checkpoint_dir.mkdir(exist_ok=True, parents=True)

   # Initialize history
   history = {
      'train_loss': [],
      'val_loss': [],
      'train_acc': [],
      'val_acc': []
   }

   best_val_loss = float('inf')
   best_model_path = checkpoint_dir / 'best_model.pth'
   epochs_without_improvement = 0

   for epoch in range(num_epochs):
      start_time = time.time()
      print(f"Epoch {epoch + 1}/{num_epochs}")

      # Train for one epoch
      train_loss, train_acc = train_epoch(
          model, train_loader, criterion, optimizer, device
      )

      # Evaluate on validation set
      val_loss, val_acc = evaluate(
          model, val_loader, criterion, device
      )

      # Update learing rate scheduler if provided
      if scheduler:
          scheduler.step(val_loss)
      
      # Save history
      history['train_loss'].append(train_loss)
      history['val_loss'].append(val_loss)
      history['train_acc'].append(train_acc)
      history['val_acc'].append(val_acc)
      print(f"Epoch {epoch + 1} completed in {time.time() - start_time:.2f} seconds")
      print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
      print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

      # Save best model
      if val_loss < best_val_loss:
         best_val_loss = val_loss
         torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc
         }, best_model_path)
         print(f"Best model saved at {best_model_path}")
         epochs_without_improvement = 0
      else:
         epochs_without_improvement += 1
         print(f"No improvement in validation loss for {epochs_without_improvement} epochs")
      
      # Checkpoint at 10 epochs
      if (epoch + 1) % 10 == 0:
         checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'
         torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc
         }, checkpoint_path)
         print(f"Checkpoint saved at {checkpoint_path}")
      
      # Early stopping
      if epochs_without_improvement >= early_stopping_patience:
         print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement")
         break
      
      print("Training completed.")
      return history, str(best_model_path)
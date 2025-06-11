import torch
from torch import nn
from torch.utils.data import DataLoader
from kitchen20 import esc
import kitchen20.utils as U
from torchaudio import transforms
input_length = 48000  # Length of audio input in samples

def main():
    # Get training and validation datasets
    k_20_transforms = []
    k_20_transforms += [U.padding(input_length // 2)]
    k_20_transforms += [U.random_crop(input_length)]
    k_20_transforms += [U.normalize(float(2**16 / 2))]
    k_20_transforms += [U.random_flip()]


    k_20_train = esc.Kitchen20(
        folds=[1, 2, 3, 4],
        audio_rate=16000,
        overwrite=False,
        transforms=k_20_transforms,
        use_bc_learning=False
    )
    k_20_val = esc.Kitchen20(
        folds=[5],
        audio_rate=16000,
        overwrite=False,
        transforms=k_20_transforms,
        use_bc_learning=False
    )

    # Create data loaders
    train_loader = DataLoader(
        k_20_train,
        batch_size=2,
        shuffle=True,
    )

    val_loader = DataLoader(
        k_20_val,
        batch_size=2,
        shuffle=False,
    )

    # Define a simple model
    model = nn.Sequential(
        nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(32 * 6000, 128),  # Adjust input size based on your data
        nn.ReLU(),
        nn.Linear(128, len(k_20_train.classes))
    )
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    num_epochs = 5
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            print(batch)
            inputs, labels = batch
            inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        print(f'Validation Loss: {val_loss:.4f}')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        scheduler.step()

if __name__ == '__main__':
    main()
# This script trains a simple CNN on the Kitchen20 dataset using PyTorch.
# It includes data loading, model definition, training, and validation loops.

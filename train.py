from model import YOLOv11mLSTMClassifier
import torch
import torch.nn as nn
import torch.optim as optim
# import torchvision
from torchvision import datasets, transforms
import os
import copy
import time
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomResizedCrop(78, scale=(0.8, 1.0)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data_dir = 'classification'  # dataset path
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                              shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes


model = YOLOv11mLSTMClassifier(num_classes=2).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    patience=3,
    factor=0.1
)

def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Initialize per-epoch metrics
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0

        # Training phase
        model.train()
        for inputs, labels in tqdm(dataloaders['train'], desc='Training'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)

        # Calculate training metrics
        epoch_train_loss = train_loss / dataset_sizes['train']
        epoch_train_acc = train_corrects.double() / dataset_sizes['train']
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        # Validation phase
        model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(dataloaders['val'], desc='Validation'):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        # Calculate validation metrics
        epoch_val_loss = val_loss / dataset_sizes['val']
        epoch_val_acc = val_corrects.double() / dataset_sizes['val']
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        # Step the scheduler with validation loss (you could use accuracy instead)
        scheduler.step(epoch_val_loss)  # Monitoring validation loss
        
        print(f'Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}')
        print(f'  Val Loss: {epoch_val_loss:.4f} |   Val Acc: {epoch_val_acc:.4f}')

        # Save model checkpoints
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_model_wts,
                'val_acc': best_acc,
            }, 'classification_result/best_model.pth')
            print(f'New best model saved with val acc: {best_acc:.4f}')

        # Save last model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'val_acc': epoch_val_acc,
        }, 'classification_result/last_model.pth')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

model, history = train_model(model, criterion, optimizer, scheduler, num_epochs=30)


# torch.save(model.state_dict(), 'classification_result/blruy_classifier.pth')
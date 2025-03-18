import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bar

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data pre - processing
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load ImageNet dataset
train_dataset = datasets.ImageFolder(root='/home/mark/mywork/test_training/Imagenet2012/train', transform=train_transform)
val_dataset = datasets.ImageFolder(root='/home/mark/mywork/test_training/Imagenet2012/val', transform=val_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

# Load ResNet model
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    # Use tqdm to show training progress
    train_progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}')
    for i, (images, labels) in train_progress:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # Update progress bar with current loss
        train_progress.set_postfix({'Loss': running_loss / (i + 1)})

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        val_progress = tqdm(val_loader, total=len(val_loader), desc='Validation')
        for images, labels in val_progress:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Update progress bar with current accuracy
            val_progress.set_postfix({'Accuracy': 100 * correct / total})

    print(f'Validation Accuracy: {100 * correct / total}%')

print('Training finished.')    

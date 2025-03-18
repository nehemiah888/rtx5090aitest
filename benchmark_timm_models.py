import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import time
from tqdm import tqdm  # Added for progress indication

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data pre - processing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load ImageNet dataset
train_dataset = datasets.ImageFolder(root='/home/mark/mywork/test_training/Imagenet2012/train', transform=transform)
val_dataset = datasets.ImageFolder(root='/home/mark/mywork/test_training/Imagenet2012/val', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

models_to_benchmark = ['vgg16', 'resnet50', 'swin_base_patch4_window7_224', 'efficientvit_m4']

for model_name in models_to_benchmark:
    print(f"Benchmarking {model_name}...")
    # FP32
    model = timm.create_model(model_name, pretrained=False, num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training benchmark FP32
    start_time = time.time()
    model.train()
    for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"{model_name} FP32 Training"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    end_time = time.time()
    train_speed_fp32 = len(train_dataset) / (end_time - start_time)
    print(f"{model_name} FP32 Training speed: {train_speed_fp32} images/second")

    # Inference benchmark FP32
    model = timm.create_model(model_name, pretrained=True, num_classes=len(train_dataset.classes)).to(device)
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(val_loader, total=len(val_loader), desc=f"{model_name} FP32 Inference"):
            images = images.to(device)
            outputs = model(images)
    end_time = time.time()
    inference_speed_fp32 = len(val_dataset) / (end_time - start_time)
    print(f"{model_name} FP32 Inference speed: {inference_speed_fp32} images/second")

    # FP16
    model = timm.create_model(model_name, pretrained=False, num_classes=len(train_dataset.classes)).to(device).half()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training benchmark FP16
    start_time = time.time()
    model.train()
    for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"{model_name} FP16 Training"):
        images = images.to(device).half()
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    end_time = time.time()
    train_speed_fp16 = len(train_dataset) / (end_time - start_time)
    print(f"{model_name} FP16 Training speed: {train_speed_fp16} images/second")

    # Inference benchmark FP16
    model = timm.create_model(model_name, pretrained=True, num_classes=len(train_dataset.classes)).to(device).half()
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(val_loader, total=len(val_loader), desc=f"{model_name} FP16 Inference"):
            images = images.to(device).half()
            outputs = model(images)
    end_time = time.time()
    inference_speed_fp16 = len(val_dataset) / (end_time - start_time)
    print(f"{model_name} FP16 Inference speed: {inference_speed_fp16} images/second")

print("Benchmarking finished.")

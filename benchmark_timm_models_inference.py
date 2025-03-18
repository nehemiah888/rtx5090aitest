import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import time
from tqdm import tqdm

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
val_dataset = datasets.ImageFolder(root='/home/mark/mywork/test_training/Imagenet2012/val', transform=transform)

# Create data loaders
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

# Move 1/3 of validation data to GPU
subset_size = len(val_dataset) // 3
all_images = []
all_labels = []
for i, (images, labels) in enumerate(tqdm(val_loader, desc="Moving 1/3 data to GPU")):
    if i * 256 >= subset_size:
        break
    all_images.append(images.to(device))
    all_labels.append(labels.to(device))
all_images = torch.cat(all_images, dim=0)
all_labels = torch.cat(all_labels, dim=0)

models_to_benchmark = ['vgg16', 'resnet50', 'tf_efficientnetv2_b0', 'swin_base_patch4_window7_224', 'efficientvit_m4']

for model_name in models_to_benchmark:
    print(f"Benchmarking {model_name} inference...")
    # FP32
    model = timm.create_model(model_name, pretrained=True, num_classes=len(val_dataset.classes)).to(device)
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(all_images), 256), desc=f"{model_name} FP32 Inference"):
            images = all_images[i:i + 256]
            outputs = model(images)
    end_time = time.time()
    # Adjust the inference speed calculation based on the subset size
    inference_speed_fp32 = subset_size / (end_time - start_time)
    print(f"{model_name} FP32 Inference speed: {inference_speed_fp32} images/second")

    # FP16
    model = timm.create_model(model_name, pretrained=True, num_classes=len(val_dataset.classes)).to(device).half()
    model.eval()
    all_images_half = all_images.half()
    start_time = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(all_images_half), 256), desc=f"{model_name} FP16 Inference"):
            images = all_images_half[i:i + 256]
            outputs = model(images)
    end_time = time.time()
    # Adjust the inference speed calculation based on the subset size
    inference_speed_fp16 = subset_size / (end_time - start_time)
    print(f"{model_name} FP16 Inference speed: {inference_speed_fp16} images/second")

print("Inference benchmarking finished.")

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
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load ImageNet dataset
train_dataset = datasets.ImageFolder(root='/home/mark/mywork/test_training/Imagenet2012/train', transform=transform)

all_models = ['vgg16', 'resnet50', 'tf_efficientnetv2_b0', 'swin_base_patch4_window7_224', 'efficientvit_m4']
print("Available models for training:")
for idx, model in enumerate(all_models, start=1):
    print(f"{idx}. {model}")
print(f"{len(all_models) + 1}. All models")
model_choice = input("Please enter the number(s) of the model(s) you want to train, separated by commas if multiple: ")
model_choice = [int(x) for x in model_choice.split(',')]
if len(all_models) + 1 in model_choice:
    models_to_benchmark = all_models
else:
    models_to_benchmark = [all_models[i - 1] for i in model_choice]

print("Available precisions for training:")
print("1. FP32")
print("2. FP16")
print("3. Both")
precision_choice = int(input("Please enter the number of the precision type you want to use: "))

# Add a prompt to select cache or not
cache_choice = input("Do you want to use cache? (y/n): ")
use_cache = cache_choice.lower() == 'y'


if precision_choice == 1:
    precisions = ['fp32']
elif precision_choice == 2:
    precisions = ['fp16']
else:
    precisions = ['fp32', 'fp16']

for model_name in models_to_benchmark:
    print(f"Benchmarking {model_name} training...")

    batch_size = 256

    # if model name is swin_base_patch4_window7_224, then reduce batch size to 128
    if model_name == 'swin_base_patch4_window7_224':
        batch_size = 128


    if 'fp32' in precisions:

        model = timm.create_model(model_name, pretrained=False, num_classes=len(train_dataset.classes)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
        # Training benchmark FP32
        
        model.train()
        if use_cache:
            cached_data = []
            num_samples = len(train_dataset)
            if model_name == 'swin_base_patch4_window7_224':
                ten_percent_count = 512
            else:
                ten_percent_count = 5120
            print("Caching 10% of the dataset...")
            for i, (img, label) in tqdm(enumerate(train_dataset), total=ten_percent_count, desc='Caching data'):
                if i >= ten_percent_count:
                    break
                cached_data.append((img, label))

            # Move cached data to GPU
            cached_data_gpu = [(img.to(device), torch.tensor(label).to(device)) for img, label in cached_data]
            cached_loader = DataLoader(cached_data_gpu, batch_size, shuffle=True, num_workers=0, pin_memory=False)

            start_time = time.time()
            for _ in range(20):
                for i, (images, labels) in tqdm(enumerate(cached_loader), total=len(cached_loader), desc=f"{model_name} FP32 Training (Cached)"):
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8)
            # Cache the first 20 iterations of data
            cached_data = []
            for i, (images, labels) in enumerate(train_loader):
                if i >= 20:
                    break
                cached_data.append((images, labels))
            start_time = time.time()
            for _ in range(20):
                for images, labels in tqdm(cached_data, desc=f"{model_name} FP32 Training"):
                    images = images.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
        end_time = time.time()
        # Adjust the training speed calculation
        if use_cache:
            train_speed_fp32 = len(cached_data) * 20 / (end_time - start_time)
        else:
            train_speed_fp32 = batch_size * 20 * 20 / (end_time - start_time)
        print(f"{model_name} FP32 Training speed: {train_speed_fp32} images/second")

    if 'fp16' in precisions:
        model = timm.create_model(model_name, pretrained=False, num_classes=len(train_dataset.classes)).to(device).half()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
        # Training benchmark FP16
        model.train()
        if use_cache:
            # Convert cached images to FP16 before the loop
            cached_data = []
            num_samples = len(train_dataset)
            ten_percent_count = 5120
            print("Caching 10% of the dataset...")
            for i, (img, label) in tqdm(enumerate(train_dataset), total=ten_percent_count, desc='Caching data'):
                if i >= ten_percent_count:
                    break
                cached_data.append((img, label))

            # Move cached data to GPU
            cached_data_gpu = [(img.to(device), torch.tensor(label).to(device)) for img, label in cached_data]
            cached_data_gpu_fp16 = [(img.half(), label) for img, label in cached_data_gpu]
            cached_loader = DataLoader(cached_data_gpu_fp16, batch_size, shuffle=True, num_workers=0, pin_memory=False)
            start_time = time.time()
            for _ in range(20):
                for i, (images, labels) in tqdm(enumerate(cached_loader), total=len(cached_loader), desc=f"{model_name} FP16 Training (Cached)"):
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8)
            # Cache the first 20 iterations of data
            cached_data = []
            for i, (images, labels) in enumerate(train_loader):
                if i >= 20:
                    break
                cached_data.append((images, labels))
            start_time = time.time()
            for _ in range(20):
                for images, labels in tqdm(cached_data, desc=f"{model_name} FP16 Training"):
                    images = images.to(device).half()
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
        end_time = time.time()
        # Adjust the training speed calculation for FP16
        if use_cache:
            train_speed_fp16 = len(cached_data) * 20 / (end_time - start_time)
        else:
            train_speed_fp16 = batch_size * 20 * 20 / (end_time - start_time)
        print(f"{model_name} FP16 Training speed: {train_speed_fp16} images/second")

print("Training benchmarking finished.")

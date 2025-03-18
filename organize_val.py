import os
import shutil

# Path to the validation images and the labels file
val_dir = 'val'
val_labels_file = 'validation_label.txt'
synset_words_file = 'train_label.txt'

# Create a mapping from class index to synset ID
class_index_to_synset = {}
with open(synset_words_file) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            image_info = parts[0].split('/')
            if len(image_info) == 2:
                synset_id = image_info[0]
                class_index = int(parts[1])
                class_index_to_synset[class_index] = synset_id

# Create a mapping from image name to synset ID using val_labels_file
image_to_synset = {}
with open(val_labels_file) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            image_name = parts[0]
            class_index = int(parts[1])
            if class_index in class_index_to_synset:
                synset = class_index_to_synset[class_index]
                image_to_synset[image_name] = synset

# Create sub - directories for each class in the validation set
synsets = set(image_to_synset.values())
for synset in synsets:
    os.makedirs(os.path.join(val_dir, synset), exist_ok=True)

# Move the validation images to the appropriate sub - directories
val_images = os.listdir(val_dir)
for image in val_images:
    if image in image_to_synset:
        synset = image_to_synset[image]
        src = os.path.join(val_dir, image)
        dst = os.path.join(val_dir, synset, image)
        shutil.move(src, dst)


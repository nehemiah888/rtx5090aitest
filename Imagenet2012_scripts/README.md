# Imagenet2012_scripts Directory

This directory contains scripts that are specifically designed to handle and preprocess the ImageNet2012 dataset. The following is a detailed description of each file in this directory:

## Python Files

### `organize_val.py`
- **Function**: This script organizes the validation images in the ImageNet2012 dataset. It reads two label files (`validation_label.txt` and `train_label.txt`) to create mappings between class indices, synset IDs, and image names. Then, it creates sub - directories for each class in the validation set and moves the validation images to their corresponding sub - directories.
- **Usage**: Make sure the label files are in the correct path and run the script. It will organize the validation images according to the labels.

## Bash Files

### `extract_train.sh`
- **Function**: This Bash script is used to extract individual class tar files in the `train` directory of the ImageNet2012 dataset. It loops through each tar file, creates a sub - directory for the class, extracts the tar file into the sub - directory, and optionally removes the original tar file after extraction.
- **Usage**: Run the script in the terminal. It will automatically handle the extraction process for all tar files in the `train` directory.


By using these scripts, you can efficiently preprocess the ImageNet2012 dataset for further deep learning tasks.

#!/bin/bash

# Directory where the main training tar was extracted
train_dir="train"

# Loop through each tar file in the train directory
for tar_file in $train_dir/*.tar; do
    # Get the base name of the tar file without the extension
    class_name=$(basename "$tar_file" .tar)
    # Create a sub - directory for the class
    mkdir -p "$train_dir/$class_name"
    # Extract the tar file into its sub - directory
    tar -xvf "$tar_file" -C "$train_dir/$class_name"
    # Optionally, you can remove the inner tar file after extraction
    rm "$tar_file"
done

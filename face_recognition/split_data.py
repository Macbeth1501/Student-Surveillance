import os
import shutil
import random

# Path to the folder containing your files
source_folder = r"Data\3"

# Paths to the new folders
train_folder = "train"
validate_folder = "validate"
test_folder = "test"

# Create new folders if they don't exist
for folder in [train_folder, validate_folder, test_folder]:
    os.makedirs(folder, exist_ok=True)

# Get list of files and shuffle them
files = os.listdir(source_folder)
random.shuffle(files)

# Split files based on the ratio
num_files = len(files)
train_split = int(num_files * 0.7)
validate_split = int(num_files * 0.2)

train_files = files[:train_split]
validate_files = files[train_split:train_split + validate_split]
test_files = files[train_split + validate_split:]

# Function to move files
def move_files(file_list, dest_folder):
    for file in file_list:
        shutil.move(os.path.join(source_folder, file), os.path.join(dest_folder, file))

# Move files to respective folders
move_files(train_files, train_folder)
move_files(validate_files, validate_folder)
move_files(test_files, test_folder)

print("Files successfully split into train, validate, and test folders!")

import os

# Set the folder path
folder_path = r"Data\3"

# Get a list of files in the folder
files = sorted(os.listdir(folder_path))  # Sorting ensures consistent order

# Rename files from 0 to x-1
for index, file in enumerate(files):
    file_extension = os.path.splitext(file)[1]  # Extract file extension
    new_name = f"{index}{file_extension}"
    old_path = os.path.join(folder_path, file)
    new_path = os.path.join(folder_path, new_name)
    os.rename(old_path, new_path)

print("Files have been renamed successfully!")

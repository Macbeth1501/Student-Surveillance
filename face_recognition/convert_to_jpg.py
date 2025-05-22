import os
from pillow_heif import open_heif
from PIL import Image

def convert_heic_to_jpg(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(".heic"):
            heic_path = os.path.join(directory, filename)
            jpg_path = os.path.join(directory, f"{os.path.splitext(filename)[0]}.jpg")

            try:
                heif_image = open_heif(heic_path)
                image = Image.frombytes(heif_image.mode, heif_image.size, heif_image.data, "raw", heif_image.mode)
                image.save(jpg_path, "JPEG")
                print(f"Converted {filename} to {jpg_path}")

                # Delete original HEIC file
                os.remove(heic_path)
                print(f"Deleted {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

directory_path = r"Data\0"  # Update the path
convert_heic_to_jpg(directory_path)

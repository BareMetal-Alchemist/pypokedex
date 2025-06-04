import os
import pyheif
from PIL import Image

def convert_heic_to_jpg(heic_path, output_dir):
    try:
        heif_file = pyheif.read(heic_path)
        img = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )

        output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(heic_path))[0] + ".jpg")
        img.save(output_path, "JPEG")
        print(f"Converted: {output_path}")
    except Exception as e:
        print(f"Failed to convert {heic_path}: {e}")

# Folder usage
input_folder = "dataset/background"
output_folder = "jpg_images"
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.lower().endswith(".heic"):
        convert_heic_to_jpg(os.path.join(input_folder, file), output_folder)

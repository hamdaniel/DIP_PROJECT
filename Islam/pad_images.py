import os
import math
from PIL import Image

def pad_images_in_directory(input_dir):
    # Loop through all files in the directory
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)

        # Skip directories
        if os.path.isdir(file_path):
            continue

        try:
            # Open image
            img = Image.open(file_path)
            w, h = img.size

            # Compute new dimensions
            new_w = int(math.ceil(float(w) / 32) * 32)
            new_h = int(math.ceil(float(h) / 32) * 32)

            # If already divisible by 32, skip padding
            if new_w == w and new_h == h:
                print("Already padded:", filename)
                continue

            # Create padded image
            padded = Image.new(img.mode, (new_w, new_h))
            padded.paste(img, (0, 0))

            # Save padded image over the original
            padded.save(file_path)

            print("Replaced with padded version:", filename)
        except Exception as e:
            print("Failed to process", filename, ":", str(e))

# Example usage
input_directory = '../datasets/BSD500_padded'  # Replace with your actual folder
pad_images_in_directory(input_directory)

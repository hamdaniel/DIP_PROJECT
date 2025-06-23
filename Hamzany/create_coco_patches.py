import os
import random
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
input_dir = "../datasets/coco"         # Folder containing coco images
output_dir = "../datasets/coco_patches"       # Where to save patches
num_patches = 5000                  # How many patches to generate
patch_width = 512
patch_height = 352
min_image_size = (patch_width, patch_height)

os.makedirs(output_dir, exist_ok=True)

# --- Load all image paths ---
image_paths = [
    os.path.join(input_dir, fname)
    for fname in os.listdir(input_dir)
    if fname.lower().endswith((".png", ".jpg", ".jpeg"))
]

if not image_paths:
    raise RuntimeError("No images found in input directory.")

# --- Patch extraction loop ---
count = 0
attempts = 0
max_attempts = num_patches * 10  # safety to avoid infinite loop

while count < num_patches and attempts < max_attempts:
    attempts += 1
    image_path = random.choice(image_paths)
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        continue

    w, h = image.size
    if w < patch_width or h < patch_height:
        continue  # skip images that are too small

    x = random.randint(0, w - patch_width)
    y = random.randint(0, h - patch_height)
    patch = image.crop((x, y, x + patch_width, y + patch_height))

    patch_path = os.path.join(output_dir, "patch_{:05d}.png".format(count))
    patch.save(patch_path)
    count += 1

    if count % 100 == 0 or count == num_patches:
        print("{count}/{num_patches} patches saved.".format(count=count, num_patches=num_patches))

print("Done: {count} patches saved to '{output_dir}'".format(count=count, output_dir=output_dir))

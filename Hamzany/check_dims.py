import os
from PIL import Image

def check_flipped_dimensions(image_dir):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No images found in directory.")
        return

    first_image_path = os.path.join(image_dir, image_files[0])
    with Image.open(first_image_path) as img:
        base_size = img.size  # (width, height)
        flipped_size = (base_size[1], base_size[0])

    print("Base size (from first image):", base_size)

    match_base = 0
    match_flipped = 0
    mismatch = []

    for fname in image_files:
        path = os.path.join(image_dir, fname)
        try:
            with Image.open(path) as img:
                size = img.size
                if size == base_size:
                    match_base += 1
                elif size == flipped_size:
                    match_flipped += 1
                else:
                    mismatch.append((fname, size))
        except Exception as e:
            print("Error reading", fname, ":", e)

    print("Matched base size:", match_base)
    print("Matched flipped size:", match_flipped)

    if mismatch:
        print("Images with mismatched sizes:")
        for fname, size in mismatch:
            print(" -", fname, ":", size)
    else:
        print("✅ All images are either base or flipped size.")

# Example usage
check_flipped_dimensions("../datasets/BSD500_padded")

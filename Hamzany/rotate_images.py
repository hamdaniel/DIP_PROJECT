import os
from PIL import Image

def rotate_to_landscape(image_dir):
    rotated_count = 0
    total_count = 0

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            image_path = os.path.join(image_dir, filename)
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    total_count += 1

                    # Rotate if height > width
                    if height > width:
                        img = img.rotate(90, expand=True)
                        img.save(image_path)
                        rotated_count += 1

            except Exception as e:
                print("❌ Failed to process image:", filename, "-", str(e))

    print("✅ Processed %d images." % total_count)
    print("🔁 Rotated %d images to landscape orientation." % rotated_count)

# Example usage
image_dir = "../datasets/BSD500_padded"
rotate_to_landscape(image_dir)

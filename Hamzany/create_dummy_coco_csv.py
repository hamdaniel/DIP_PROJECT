import os
import csv

patches_dir = "../datasets/coco_patches"  # same as your output_dir
csv_path = "../datasets/coco_patches/dummy_data.csv"  # path to save CSV

# List all image files in the patches directory
image_files = sorted([
    f for f in os.listdir(patches_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

# Define CSV header
header = ['image','iter_num', 'time']

with open(csv_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

    for img_name in image_files:
        for i in range(16):
            # Each image will have 16 rows with iter_num and time as zeros
            # 'iter_num' and 'time' are placeholders, can be replaced with actual values later
            # Here we just use zeros as dummy values
        # Get image filename without extension
        
        # Row: image ID + sixteen zeros
            row = [img_name, i, 0.0]
            writer.writerow(row)

print("CSV saved to: {csv_path}".format(csv_path=csv_path))

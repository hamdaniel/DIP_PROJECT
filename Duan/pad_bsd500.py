import os
from PIL import Image
import argparse
import math

def round_up_to_multiple(x, base):
    return int(math.ceil(x / base) * base)

def resize_image_to_divisible(im, divisor):
    w, h = im.size
    new_w = round_up_to_multiple(w, divisor)
    new_h = round_up_to_multiple(h, divisor)
    if (w, h) == (new_w, new_h):
        return im
    return im.resize((new_w, new_h), Image.LANCZOS)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="../datasets/BSD500/val" ,help="Path to the image directory (non-recursive)")
    parser.add_argument("--divisor", type=int, default=64, help="Make height and width divisible by this number")
    args = parser.parse_args()

    input_dir = args.directory
    divisor = args.divisor
    output_dir = f"{input_dir}_resized_div{divisor}"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        try:
            with Image.open(input_path) as im:
                im = resize_image_to_divisible(im, divisor)
                im.save(output_path)
        except Exception as e:
            print(f"Skipping {filename} (not an image or failed): {e}")

    print(f"Done. Resized images saved in: {output_dir}")

if __name__ == "__main__":
    main()

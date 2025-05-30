#!/bin/bash

mkdir -p BSD500/codes_cpu
mkdir -p BSD500/decoded_cpu
mkdir -p BSD500/val_padded_cpu

# Collect all image paths matching .png or .jpg
shopt -s nullglob
image_files=(../datasets/BSD500/val/*.{png,jpg})
shopt -u nullglob

total=${#image_files[@]}
if (( total == 0 )); then
  echo "No images found in ../datasets/BSD500/val/"
  exit 1
fi

count=0
for img_path in "${image_files[@]}"; do
  ((count++))

  filename=$(basename "$img_path")
  name="${filename%.*}"

  padded_img="BSD500/val_padded/${name}_padded.png"

  echo "[$count/$total] Padding $filename"
  python3 -c "
from PIL import Image
import math
img = Image.open('$img_path')
w, h = img.size
new_w = math.ceil(w / 32) * 32
new_h = math.ceil(h / 32) * 32
padded = Image.new(img.mode, (new_w, new_h))
padded.paste(img, (0, 0))
padded.save('$padded_img')
"

  echo "[$count/$total] Encoding $padded_img"
  python3 encoder.py --model checkpoint/encoder_epoch_00000001.pth --input "$padded_img" --cuda --output "BSD500/codes/$name" --iterations 16

  echo "[$count/$total] Decoding BSD500/codes/$name.npz"
  mkdir -p "BSD500/decoded/$name"
  python3 decoder.py --model checkpoint/decoder_epoch_00000001.pth --input "BSD500/codes/$name.npz" --cuda --output "BSD500/decoded/$name"
done

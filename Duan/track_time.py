import os
import time
import csv
import torch
import torchvision.transforms.functional as tvf
from PIL import Image
import math
from lvae import get_model
torch.set_grad_enabled(False)
# Assuming your model loading function is already defined
# Example placeholder:
# def get_model(name, lmb, pretrained): ...

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_progressive_coding(model, img_path):
    im = tvf.to_tensor(Image.open(img_path)).unsqueeze(0).to(device=device)
    nB, imC, imH, imW = im.shape

    # Measure encoder time
    encoder_start = time.time()
    stats_all = model.forward_get_latents(im)
    encoder_time = time.time() - encoder_start

    progressive_decodings = []
    timings = []

    L = len(stats_all)
    for keep in range(1, L + 1):
        start_time = time.time()

        latents = [stat['z'] if (i < keep) else None for (i, stat) in enumerate(stats_all)]
        kl_divs = [stat['kl'] for (i, stat) in enumerate(stats_all) if (i < keep)]
        kl = sum([kl.sum(dim=(1, 2, 3)) for kl in kl_divs]) / (imH * imW) * math.log2(math.e)

        sample = model.cond_sample(latents, temprature=0)
        progressive_decodings.append(sample.squeeze(0))

        decode_time = time.time() - start_time
        total_time = encoder_time + decode_time
        timings.append(total_time)

        print(f'Keep={keep}, bpp={kl.item():.4f}, encode+decode_time={total_time:.4f}s (encode={encoder_time:.4f}, decode={decode_time:.4f})')

    return torch.stack(progressive_decodings, dim=0), timings

def process_directory(img_dir, model, output_csv_path):
    rows = []

    for filename in os.listdir(img_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(img_dir, filename)
            print(f'\nProcessing {filename}...')
            _, timings = run_progressive_coding(model, img_path)
            row = [os.path.splitext(filename)[0]] + timings
            rows.append(row)

    max_latents = max(len(r) - 1 for r in rows)
    header = ['image_name'] + [f'time_for_{i + 1}_latent' for i in range(max_latents)]

    with open(output_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            padded_row = row + [''] * (max_latents + 1 - len(row))
            writer.writerow(padded_row)

    print(f'\nTiming results saved to: {output_csv_path}')


# Main script
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run progressive coding on a directory of images.')
    parser.add_argument('--input_dir', type=str, default="../datasets/BSD500/val_resized_div64", help='Directory containing input images.')
    parser.add_argument('--output_csv', type=str, default='timings.csv', help='Output CSV file for timing data.')
    args = parser.parse_args()

    model = get_model('qres34m', lmb=2048, pretrained=True)
    model = model.to(device=device).eval()

    process_directory(args.input_dir, model, args.output_csv)

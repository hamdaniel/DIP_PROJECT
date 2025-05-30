import argparse
import os
import math
import io
from PIL import Image
import torch
import torchvision.transforms.functional as tvf
from lvae import get_model
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def run_and_return_bpp(model, img_path):
    im = tvf.to_tensor(Image.open(img_path)).unsqueeze(0).to(device=device)
    _, _, imH, imW = im.shape

    stats_all = model.forward_get_latents(im)
    latents = [stat['z'] for stat in stats_all]
    kl_divs = [stat['kl'] for stat in stats_all]
    kl = sum([kl.sum(dim=(1, 2, 3)) for kl in kl_divs]) / (imH * imW) * math.log2(math.e)
    bpp = kl.item()
    sample = model.cond_sample(latents, temprature=0).squeeze(0).cpu().clamp(0, 1)
    return sample, bpp


def run_progressive_coding(model, img_path):
    im = tvf.to_tensor(Image.open(img_path)).unsqueeze(0).to(device=device)
    nB, imC, imH, imW = im.shape

    stats_all = model.forward_get_latents(im)
    L = len(stats_all)

    latents = [stat['z'] for stat in stats_all]
    kl_divs = [stat['kl'] for stat in stats_all]
    kl = sum([kl.sum(dim=(1, 2, 3)) for kl in kl_divs]) / (imH * imW) * math.log2(math.e)
    sample = model.cond_sample(latents, temprature=0).squeeze(0)

    return sample, kl.item()


def jpeg_compress(img_path, quality):
    """Compress an image using JPEG with specified quality and return tensor and bpp"""
    # Open original image
    original_img = Image.open(img_path)
    
    # Save with JPEG compression to a BytesIO object to get file size
    buffer = io.BytesIO()
    original_img.save(buffer, format="JPEG", quality=quality)
    compressed_size = buffer.tell()  # Size in bytes
    
    # Get compressed image as tensor
    buffer.seek(0)
    compressed_img = Image.open(buffer)
    compressed_tensor = tvf.to_tensor(compressed_img)
    
    # Calculate bpp
    width, height = original_img.size
    pixel_count = width * height
    bpp = (compressed_size * 8) / pixel_count
    
    return compressed_tensor, bpp


def save_comparison_plot(original_tensor, recon_tensors, bpps, jpeg_tensors, jpeg_bpps, jpeg_qualities, out_path, lambdas):
    # Determine number of plots for model reconstructions
    n_model = len(recon_tensors)
    half = math.ceil(n_model / 2)
    
    # Create a layout with 3 rows
    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    
    # Turn off all axes
    for ax in axs.flat:
        ax.axis('off')
    
    # First row: Original and JPEG reconstructions
    axs[0, 0].imshow(tvf.to_pil_image(original_tensor.cpu().clamp(0, 1)))
    axs[0, 0].set_title("Original")
    
    for i, (tensor, bpp, quality) in enumerate(zip(jpeg_tensors, jpeg_bpps, jpeg_qualities)):
        axs[0, i+1].imshow(tvf.to_pil_image(tensor))
        axs[0, i+1].set_title(f"JPEG q={quality}\n{bpp:.3f} bpp")
    
    # Second row: first half of QRes34 reconstructions
    for i in range(4):
        axs[1, i].imshow(tvf.to_pil_image(recon_tensors[i].cpu().clamp(0, 1)))
        axs[1, i].set_title(f"QRes34 λ={lambdas[i]}\n{bpps[i]:.3f} bpp")
    
    # Third row: second half of QRes34 reconstructions
    for i in range(4):
        axs[2, i].imshow(tvf.to_pil_image(recon_tensors[i+4].cpu().clamp(0, 1)))
        axs[2, i].set_title(f"QRes34 λ={lambdas[i+4]}\n{bpps[i+4]:.3f} bpp")
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    allowed_lambdas = [16, 32, 64, 128, 256, 512, 1024, 2048]
    parser = argparse.ArgumentParser(description='Run QRes34 model and JPEG compression for comparison.')
    parser.add_argument('--lmbs', type=int, nargs='+', default=allowed_lambdas, help='List of lambda values for QRes34')
    parser.add_argument('--images', nargs='*', default=None, help='Paths to input images (optional)')
    parser.add_argument('--outdir', type=str, default='./outputs_with_jpeg', help='Directory to save outputs')
    args = parser.parse_args()

    # Define JPEG quality levels (low, medium, high)
    jpeg_qualities = [10, 50, 90]

    for lmb in args.lmbs:
        if lmb not in allowed_lambdas:
            raise ValueError(f"Invalid lambda: {lmb}. Allowed values are: {sorted(allowed_lambdas)}")

    if args.images is None or len(args.images) == 0:
        # Default to all images in ./../datasets/kodak folder
        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
        images = []
        for ext in image_extensions:
            images.extend(glob(os.path.join('../datasets/kodak', ext)))
        if not images:
            print("No images found in ./../datasets/kodak folder.")
            return
    else:
        images = args.images

    os.makedirs(args.outdir, exist_ok=True)

    # Load all models once
    models = {}
    for lmb in args.lmbs:
        print(f'Loading model for λ={lmb}...')
        model = get_model('qres34m', lmb=lmb, pretrained=True).to(device).eval()
        models[lmb] = model

    # Dictionaries to accumulate metrics
    bpp_accum = {lmb: [] for lmb in args.lmbs}
    psnr_accum = {lmb: [] for lmb in args.lmbs}
    jpeg_bpp_accum = {q: [] for q in jpeg_qualities}
    jpeg_psnr_accum = {q: [] for q in jpeg_qualities}

    for img_path in images:
        # Load and prepare original image
        original = tvf.to_tensor(Image.open(img_path)).clamp(0, 1).to(device)

        # Sort lambdas ascending
        sorted_lmbs = sorted(args.lmbs)

        # Collect reconstructions and bpps for QRes34 model
        model_reconstructions = []
        model_bpps = []
        for lmb in sorted_lmbs:
            model = models[lmb]
            output, bpp = run_and_return_bpp(model, img_path)
            output = output.to(device)
            model_reconstructions.append(output.cpu())
            model_bpps.append(bpp)

            # Compute PSNR for this reconstruction
            psnr_val = compute_psnr(original.cpu(), output.cpu())
            bpp_accum[lmb].append(bpp)
            psnr_accum[lmb].append(psnr_val.item())

        # Collect JPEG reconstructions and bpps
        jpeg_reconstructions = []
        jpeg_bpps = []
        for quality in jpeg_qualities:
            jpeg_img, jpeg_bpp = jpeg_compress(img_path, quality)
            jpeg_reconstructions.append(jpeg_img)
            jpeg_bpps.append(jpeg_bpp)
            
            # Compute PSNR for JPEG
            jpeg_psnr = compute_psnr(original.cpu(), jpeg_img.to(device).cpu())
            jpeg_bpp_accum[quality].append(jpeg_bpp)
            jpeg_psnr_accum[quality].append(jpeg_psnr.item())

        # Save comparison plot
        base_name = os.path.basename(img_path)
        fig_path = os.path.join(args.outdir, f'compare_with_jpeg_{base_name}')
        save_comparison_plot(original, model_reconstructions, model_bpps, 
                           jpeg_reconstructions, jpeg_bpps, jpeg_qualities,
                           fig_path, sorted_lmbs)
        print(f"Saved comparison plot to {fig_path}")

    # Generate RD Plot
    plt.figure(figsize=(10, 8))
    
    # Plot QRes34 model points
    model_avg_bpps = []
    model_avg_psnrs = []
    for lmb in args.lmbs:
        mean_bpp = np.mean(bpp_accum[lmb])
        mean_psnr = np.mean(psnr_accum[lmb])
        model_avg_bpps.append(mean_bpp)
        model_avg_psnrs.append(mean_psnr)
    
    # Sort by rate for plotting
    sorted_model_pairs = sorted(zip(model_avg_bpps, model_avg_psnrs, args.lmbs))
    sorted_model_bpps, sorted_model_psnrs, sorted_lmbs = zip(*sorted_model_pairs)
    
    plt.plot(sorted_model_bpps, sorted_model_psnrs, marker='o', linestyle='-', label='QRes34M', color='blue')
    for lmb, x, y in zip(sorted_lmbs, sorted_model_bpps, sorted_model_psnrs):
        plt.text(x + 0.01, y, f'λ={lmb}', fontsize=9, ha='left', va='center')

    # Plot JPEG points
    jpeg_avg_bpps = []
    jpeg_avg_psnrs = []
    for quality in jpeg_qualities:
        mean_bpp = np.mean(jpeg_bpp_accum[quality])
        mean_psnr = np.mean(jpeg_psnr_accum[quality])
        jpeg_avg_bpps.append(mean_bpp)
        jpeg_avg_psnrs.append(mean_psnr)
    
    plt.plot(jpeg_avg_bpps, jpeg_avg_psnrs, marker='s', linestyle='-', label='JPEG', color='red')
    for quality, x, y in zip(jpeg_qualities, jpeg_avg_bpps, jpeg_avg_psnrs):
        plt.text(x + 0.01, y, f'q={quality}', fontsize=9, ha='left', va='center')

    plt.xlabel('Bits Per Pixel (bpp)')
    plt.ylabel('PSNR (dB)')
    plt.title('Rate-Distortion Curve: QRes34M vs JPEG')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    rd_path = os.path.join(args.outdir, 'rate_distortion_curve.png')
    plt.savefig(rd_path)
    plt.close()
    print(f"Saved RD curve to {rd_path}")

    # Save per-image results for both methods
    per_image_csv = os.path.join(args.outdir, 'per_image_results.csv')
    with open(per_image_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Create header with model and JPEG columns
        header = ['image']
        # Add model columns
        for lmb in args.lmbs:
            header.append(f'QRes34_λ={lmb}_bpp')
        for lmb in args.lmbs:
            header.append(f'QRes34_λ={lmb}_psnr')
        # Add JPEG columns
        for quality in jpeg_qualities:
            header.append(f'JPEG_q={quality}_bpp')
        for quality in jpeg_qualities:
            header.append(f'JPEG_q={quality}_psnr')
        
        writer.writerow(header)
        
        # Add data for each image
        for idx, img_path in enumerate(images):
            row = [os.path.basename(img_path)]
            # Add model data
            for lmb in args.lmbs:
                row.append(bpp_accum[lmb][idx])
            for lmb in args.lmbs:
                row.append(psnr_accum[lmb][idx])
            # Add JPEG data
            for quality in jpeg_qualities:
                row.append(jpeg_bpp_accum[quality][idx])
            for quality in jpeg_qualities:
                row.append(jpeg_psnr_accum[quality][idx])
            
            writer.writerow(row)
    print(f"Saved per-image results to {per_image_csv}")

    # Save average bpp and PSNR values for both methods
    avg_csv = os.path.join(args.outdir, 'rd_curve.csv')
    with open(avg_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'parameter', 'avg_bpp', 'avg_psnr'])
        
        # QRes34 model data
        for lmb, bpp, psnr in zip(sorted_lmbs, sorted_model_bpps, sorted_model_psnrs):
            writer.writerow(['QRes34M', lmb, bpp, psnr])
        
        # JPEG data
        for quality, bpp, psnr in zip(jpeg_qualities, jpeg_avg_bpps, jpeg_avg_psnrs):
            writer.writerow(['JPEG', quality, bpp, psnr])
            
    print(f"Saved average RD values to {avg_csv}")

if __name__ == '__main__':
    main()
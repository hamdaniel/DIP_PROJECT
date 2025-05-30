import argparse
import os
import math
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


def save_comparison_plot(original_tensor, recon_tensors, bpps, out_path, lambdas):
    n = len(recon_tensors) + 1
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))

    # Original image
    axs[0].imshow(tvf.to_pil_image(original_tensor.cpu().clamp(0, 1)))
    axs[0].axis('off')
    axs[0].set_title("Original")

    # Reconstructed images
    for i, (tensor, bpp, lmb) in enumerate(zip(recon_tensors, bpps, lambdas), start=1):
        axs[i].imshow(tvf.to_pil_image(tensor.cpu().clamp(0, 1)))
        axs[i].axis('off')
        axs[i].set_title(f"λ={lmb}\n{bpp:.3f} bpp")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    allowed_lambdas = [16, 32, 64, 128, 256, 512, 1024, 2048]
    parser = argparse.ArgumentParser(description='Run QRes34 model progressive coding and visualize results.')
    parser.add_argument('--lmbs', type=int, nargs='+', default=allowed_lambdas, help='List of lambda values')
    parser.add_argument('--images', nargs='*', default=None, help='Paths to input images (optional)')
    parser.add_argument('--outdir', type=str, default='./outputs', help='Directory to save outputs')
    args = parser.parse_args()

    allowed_lambdas = {16, 32, 64, 128, 256, 512, 1024, 2048} # 16 32 64 128 256 512 1024 2048
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

    # Dictionaries to accumulate bpps and PSNRs
    bpp_accum = {lmb: [] for lmb in args.lmbs}
    psnr_accum = {lmb: [] for lmb in args.lmbs}

    for img_path in images:
        # Load and prepare original image
        original = tvf.to_tensor(Image.open(img_path)).clamp(0, 1).to(device)

        # Sort lambdas ascending
        sorted_lmbs = sorted(args.lmbs)

        # Collect reconstructions and bpps in ascending lambda order
        reconstructions = []
        bpps = []
        for lmb in sorted_lmbs:
            model = models[lmb]
            output, bpp = run_and_return_bpp(model, img_path)
            output = output.to(device)
            reconstructions.append(output.cpu())
            bpps.append(bpp)

            # Compute PSNR for this reconstruction
            psnr_val = compute_psnr(original.cpu(), output.cpu())
            bpp_accum[lmb].append(bpp)
            psnr_accum[lmb].append(psnr_val.item())

        # Plotting setup:
        n_recon = len(reconstructions)
        half = math.ceil(n_recon / 2)

        fig, axs = plt.subplots(3, max(half, 4), figsize=(4 * max(half, 4), 12))

        # Clear axes if total grid is bigger than needed
        for ax in axs.flat:
            ax.axis('off')

        # First row: original image, leftmost subplot only
        axs[0, 0].imshow(tvf.to_pil_image(original.cpu()))
        axs[0, 0].set_title("Original")
        axs[0, 0].axis('off')

        # If more than one subplot in first row, leave others blank (or hide)
        for ax in axs[0, 1:]:
            ax.axis('off')

        # Second row: first half of reconstructed images
        for i in range(half):
            axs[1, i].imshow(tvf.to_pil_image(reconstructions[i]))
            axs[1, i].set_title(f"λ={sorted_lmbs[i]}\n{bpps[i]:.3f} bpp")
            axs[1, i].axis('off')

        # Third row: second half of reconstructed images
        for i in range(half, n_recon):
            axs[2, i - half].imshow(tvf.to_pil_image(reconstructions[i]))
            axs[2, i - half].set_title(f"λ={sorted_lmbs[i]}\n{bpps[i]:.3f} bpp")
            axs[2, i - half].axis('off')

        base_name = os.path.basename(img_path)
        fig_path = os.path.join(args.outdir, f'compare_{base_name}')
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved comparison plot to {fig_path}")

    # Generate RD Plot
    avg_bpps = []
    avg_psnrs = []
    for lmb in args.lmbs:
        mean_bpp = np.mean(bpp_accum[lmb])
        mean_psnr = np.mean(psnr_accum[lmb])
        avg_bpps.append(mean_bpp)
        avg_psnrs.append(mean_psnr)

    # Sort by rate
    sorted_pairs = sorted(zip(avg_bpps, avg_psnrs, args.lmbs))
    sorted_bpps, sorted_psnrs, sorted_lmbs = zip(*sorted_pairs)

    plt.figure(figsize=(8, 6))
    plt.plot(sorted_bpps, sorted_psnrs, marker='o', linestyle='-')
    for lmb, x, y in zip(sorted_lmbs, sorted_bpps, sorted_psnrs):
        plt.text(x - 0.01, y - 0.3, f'{lmb}', fontsize=9, ha='left', va='top')  # shifted right and down
    plt.xlabel('Bits Per Pixel (bpp)')
    plt.ylabel('PSNR (dB)')
    plt.title('Rate-Distortion Curve (QRes34M)')
    plt.grid(True, linestyle='--', alpha=0.6)
    rd_path = os.path.join(args.outdir, 'rate_distortion_curve.png')
    plt.savefig(rd_path)
    plt.close()
    print(f"Saved RD curve to {rd_path}")

    # Save per-image bpp and PSNR values
    per_image_csv = os.path.join(args.outdir, 'per_image_results.csv')
    with open(per_image_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['image'] + [f'λ={lmb}_bpp' for lmb in args.lmbs] + [f'λ={lmb}_psnr' for lmb in args.lmbs]
        writer.writerow(header)
        for idx, img_path in enumerate(images):
            row = [os.path.basename(img_path)]
            row += [bpp_accum[lmb][idx] for lmb in args.lmbs]
            row += [psnr_accum[lmb][idx] for lmb in args.lmbs]
            writer.writerow(row)
    print(f"Saved per-image results to {per_image_csv}")

    # Save average bpp and PSNR values
    avg_csv = os.path.join(args.outdir, 'rd_curve.csv')
    with open(avg_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lambda', 'avg_bpp', 'avg_psnr'])
        for lmb, bpp, psnr in zip(sorted_lmbs, sorted_bpps, sorted_psnrs):
            writer.writerow([lmb, bpp, psnr])
    print(f"Saved average RD values to {avg_csv}")

if __name__ == '__main__':
    main()

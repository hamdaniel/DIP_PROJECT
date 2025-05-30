import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
import math
import io
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.utils as tv
from lvae import get_model
import pandas as pd

def jpeg_compress(img_path, quality):
    """Compress an image using JPEG with specified quality and return tensor and bpp"""
    original_img = Image.open(img_path)
    
    buffer = io.BytesIO()
    original_img.save(buffer, format="JPEG", quality=quality)
    compressed_size = buffer.tell()
    
    buffer.seek(0)
    compressed_img = Image.open(buffer)
    compressed_tensor = tvf.to_tensor(compressed_img)
    
    width, height = original_img.size
    pixel_count = width * height
    bpp = (compressed_size * 8) / pixel_count
    
    return compressed_tensor, bpp

def run_progressive_coding(model, img_path, device='cuda'):
    with torch.no_grad():
        im_gt = tvf.to_tensor(Image.open(img_path)).unsqueeze(0).to(device=device)
        nB, imC, imH, imW = im_gt.shape

        stats_all = model.forward_get_latents(im_gt)

        progressive_decodings = []
        bpps = []
        psnrs = []

        L = len(stats_all)
        for keep in range(1, L+1):
            latents = [stat['z'] if (i < keep) else None for (i, stat) in enumerate(stats_all)]
            kl_divs = [stat['kl'] for (i, stat) in enumerate(stats_all) if (i < keep)]
            kl = sum([kl.sum(dim=(1,2,3)) for kl in kl_divs]) / (imH * imW) * math.log2(math.e)

            sample = model.cond_sample(latents, temprature=0).clamp(0, 1)
            psnr = 10 * torch.log10(1.0 / F.mse_loss(sample, im_gt))

            progressive_decodings.append(sample.squeeze(0).cpu())
            bpps.append(kl.item())
            psnrs.append(psnr.item())

            print(f'Keep={keep}, bpp={kl.item():.4f}, PSNR={psnr.item():.2f} dB')

        return torch.stack(progressive_decodings), bpps, psnrs

def plot_labeled_grid(all_recons, lambdas, jpeg_recons, jpeg_qualities, save_path=None):
    import matplotlib.patches as patches

    all_recons_cat = torch.cat(all_recons, dim=0)
    nrow = all_recons[0].shape[0]  # progressive steps per row (columns)
    ncol = len(lambdas)  # lambdas are rows, steps are columns

    # Create figure with extra row for JPEG
    fig, ax = plt.subplots(nrows=len(lambdas) + 1, ncols=nrow, figsize=(16, 2.5 * (len(lambdas) + 1)))
    
    plt.subplots_adjust(left=0.05, right=0.99, top=0.90, bottom=0.05, wspace=0.1, hspace=0.5)

    
    fig.suptitle("Progressive Decoding Reconstructions", fontsize=30, fontweight='bold')

    # Plot JPEG reconstructions in the first row
    ax[0, 0].annotate(
        'JPEG',
        xy=(-1, 0.5),
        xycoords='axes fraction',
        fontsize=28,
        ha='right',
        va='center',
        fontweight='bold'
    )
    
    for j, (jpeg_img, quality) in enumerate(zip(jpeg_recons, jpeg_qualities)):
        ax[0, j].imshow(jpeg_img.permute(1, 2, 0).numpy())
        ax[0, j].axis('off')
        ax[0, j].set_title(f'q={quality}', fontsize=20)

    num_total_slots = ax.shape[1]
    num_used_slots = len(jpeg_qualities)

    for j in range(num_used_slots, num_total_slots):
        ax[0, j].axis('off')

    # Plot progressive reconstructions
    for i in range(len(lambdas)):
        for j in range(nrow):
            img_idx = i * nrow + j
            ax[i+1, j].imshow(all_recons_cat[img_idx].permute(1, 2, 0).numpy())
            ax[i+1, j].axis('off')

        ax[i+1, 0].annotate(
            f'λ={lambdas[i]}',
            xy=(-1, 0.5),
            xycoords='axes fraction',
            fontsize=28,
            ha='right',
            va='center',
            fontweight='bold'
        )

    for j in range(nrow):
        ax[1, j].annotate(
            f'{j+1}',
            xy=(0.5, 1.1),
            xycoords='axes fraction',
            fontsize=28,
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved labeled grid plot to {save_path}")

    plt.close(fig)

def main():
    device = 'cuda'
    img_paths = [f"../datasets/kodak/kodim{str(i).zfill(2)}.png" for i in [4, 9]]
    lambdas = [16, 32, 64, 128, 256, 512, 1024, 2048]
    jpeg_qualities = [10, 50, 90]  # Low, medium, high quality

    rd_save_dir = 'progressive_rd_plots_jpeg'
    grids_save_dir = 'progressive_grids_jpeg'
    os.makedirs(rd_save_dir, exist_ok=True)
    os.makedirs(grids_save_dir, exist_ok=True)

    for img_path in img_paths:
        all_bpps = []
        all_psnrs = []
        all_recons = []
        
        # Get JPEG reconstructions first
        jpeg_recons = []
        jpeg_bpps = []
        jpeg_psnrs = []
        original = tvf.to_tensor(Image.open(img_path))
        
        for quality in jpeg_qualities:
            jpeg_tensor, bpp = jpeg_compress(img_path, quality)
            psnr = 10 * torch.log10(1.0 / F.mse_loss(jpeg_tensor, original))
            
            jpeg_recons.append(jpeg_tensor)
            jpeg_bpps.append(bpp)
            jpeg_psnrs.append(psnr.item())
            print(f'JPEG q={quality}, bpp={bpp:.4f}, PSNR={psnr:.2f} dB')

        for lmb in lambdas:
            print(f'Running model for λ={lmb}')
            model = get_model('qres34m', lmb=lmb, pretrained=True).to(device).eval()
            ims, bpps, psnrs = run_progressive_coding(model, img_path, device)
            all_bpps.append(bpps)
            all_psnrs.append(psnrs)
            all_recons.append(ims)

        # Read external RD data CSV
        csv_path = "../Islam/my_recons/all_images_rd_values.csv"
        rd_df = pd.read_csv(csv_path)

        # Plot RD curves
        plt.figure(figsize=(10, 6))
        
        # Plot JPEG points
        plt.plot(jpeg_bpps, jpeg_psnrs, marker='^', linestyle='-', color='red', label='JPEG')
        for quality, bpp, psnr in zip(jpeg_qualities, jpeg_bpps, jpeg_psnrs):
            plt.annotate(f'q={quality}', (bpp, psnr), xytext=(5, 5), textcoords='offset points')
        
        # Plot progressive coding curves
        for i, lmb in enumerate(lambdas):
            plt.plot(all_bpps[i], all_psnrs[i], marker='o', label=f'λ={lmb}')

        # Add external data from CSV
        img_name = os.path.basename(img_path).split('.')[0]
        df_img = rd_df[rd_df['image_name'] == img_name]

        if not df_img.empty:
            plt.plot(df_img['bpp'], df_img['psnr'], 's--', color='orange', label='Islam')

        plt.xlabel('Bits per pixel (bpp)')
        plt.ylabel('PSNR (dB)')
        plt.title(f'Rate-Distortion Curves for {img_name}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(rd_save_dir, f"rd_curve_{img_name}.png"))
        plt.close()

        # Save image grid with labels
        grid_save_path = os.path.join(grids_save_dir, f"grid_{os.path.basename(img_path).split('.')[0]}.png")
        plot_labeled_grid(all_recons, lambdas, jpeg_recons, jpeg_qualities, save_path=grid_save_path)

if __name__ == "__main__":
    main()
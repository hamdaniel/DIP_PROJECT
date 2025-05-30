import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
import math
from PIL import Image
import matplotlib.pyplot as plt
from lvae import get_model
import pandas as pd

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

        return bpps, psnrs

def main():
    device = 'cuda'
    img_paths = [f"../datasets/kodak/kodim{str(i).zfill(2)}.png" for i in [4, 9]]
    lambdas = [16, 32, 64, 128, 256, 512, 1024, 2048]

    rd_save_dir = 'reverse_rd_plots'
    os.makedirs(rd_save_dir, exist_ok=True)

    for img_path in img_paths:
        all_bpps = []
        all_psnrs = []

        for lmb in lambdas:
            print(f'Running model for λ={lmb}')
            model = get_model('qres34m', lmb=lmb, pretrained=True).to(device).eval()
            bpps, psnrs = run_progressive_coding(model, img_path, device)
            all_bpps.append(bpps)
            all_psnrs.append(psnrs)

        # Read external RD data CSV once
        csv_path = "../Islam/my_recons/all_images_rd_values.csv"
        rd_df = pd.read_csv(csv_path)

        # --- Plot RD curves ---
        plt.figure(figsize=(10, 6))

        # Transpose data to group by progressive step
        num_steps = len(all_bpps[0])
        for step in range(num_steps):
            step_bpps = [all_bpps[i][step] for i in range(len(lambdas))]
            step_psnrs = [all_psnrs[i][step] for i in range(len(lambdas))]
            plt.plot(step_bpps, step_psnrs, marker='o', label=f'Step {step+1}')

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
        plt.savefig(os.path.join(rd_save_dir, f"rd_reverse_curve_{img_name}.png"))
        plt.close()

if __name__ == "__main__":
    main()

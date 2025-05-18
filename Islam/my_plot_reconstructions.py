from __future__ import division, print_function
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # So it works on servers without display
import matplotlib.pyplot as plt
from PIL import Image
import csv

def compute_bpp(npz_path, height, width):
    file_size_bits = os.path.getsize(npz_path) * 8  # bytes to bits
    return file_size_bits / (height * width)

def load_image(path):
    return np.array(Image.open(path))

def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100  # perfect match
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr

def plot_reconstructions(original_path, recon_folder, npz_path, save_path):
    try:
        original = load_image(original_path)
    except:
        print("Could not load original image:", original_path)
        return None

    height, width = original.shape[:2]

    if not os.path.exists(npz_path):
        print("Missing code file:", npz_path)
        return None

    total_bpp = compute_bpp(npz_path, height, width)

    recon_images = []
    for i in range(16):
        img_path = os.path.join(recon_folder, '%02d.png' % i)
        if not os.path.exists(img_path):
            print("Missing decoded image:", img_path)
            return None
        img = load_image(img_path)
        recon_images.append(img)

    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    plt.tight_layout()

    for j in range(4):
        axes[0, j].axis('off')
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original")

    for i in range(16):
        row = 1 + i // 4
        col = i % 4
        bpp = (total_bpp * (i + 1)) / 16
        axes[row][col].imshow(recon_images[i])
        axes[row][col].set_title("Iter %d\n%.2f bpp" % (i + 1, bpp))
        axes[row][col].axis('off')

    plt.savefig(save_path)
    plt.close()
    print("Saved:", save_path)

    # Compute PSNRs for this image, return them for aggregation
    bpp_vals = []
    psnr_vals = []
    for i in range(16):
        bpp = (total_bpp * (i + 1)) / 16
        psnr = compute_psnr(original, recon_images[i])
        bpp_vals.append(bpp)
        psnr_vals.append(psnr)

    return bpp_vals, psnr_vals

def main():
    input_folder = 'test/images'
    recon_root = 'test/decoded'
    code_root = 'test/codes'
    output_folder = 'my_recons'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_bpp = []
    all_psnr = []

    # For saving all individual RD values
    all_rd_rows = []  # will hold tuples: (image_name, iter, bpp, psnr)

    for fname in os.listdir(input_folder):
        if fname.endswith('.png'):
            img_name = os.path.splitext(fname)[0]
            original_path = os.path.join(input_folder, fname)
            recon_folder = os.path.join(recon_root, img_name)
            npz_path = os.path.join(code_root, img_name + '.npz')
            save_path = os.path.join(output_folder, img_name + '_recons.png')

            result = plot_reconstructions(original_path, recon_folder, npz_path, save_path)
            if result is None:
                continue

            bpp_vals, psnr_vals = result

            # Save all image data for CSV
            for i, (bpp, psnr) in enumerate(zip(bpp_vals, psnr_vals), start=1):
                all_rd_rows.append((img_name, i, bpp, psnr))

            if not all_bpp:
                all_bpp = bpp_vals  # assume same for all images
            all_psnr.append(psnr_vals)

    if not all_psnr:
        print("No RD data collected.")
        return

    all_psnr = np.array(all_psnr)  # shape (num_images, 16)
    avg_psnr = np.mean(all_psnr, axis=0)

    # Save average RD values to CSV
    csv_avg_path = os.path.join(output_folder, 'average_rd_values.csv')
    with open(csv_avg_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['bpp', 'avg_psnr'])
        for bpp, psnr in zip(all_bpp, avg_psnr):
            writer.writerow([bpp, psnr])
    print("Saved average RD values to CSV:", csv_avg_path)

    # Save all individual images RD values to CSV
    csv_all_path = os.path.join(output_folder, 'all_images_rd_values.csv')
    with open(csv_all_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'iter', 'bpp', 'psnr'])
        for row in all_rd_rows:
            writer.writerow(row)
    print("Saved all individual images RD values to CSV:", csv_all_path)

    # Plot average RD curve
    plt.figure(figsize=(8, 6))
    plt.plot(all_bpp, avg_psnr, marker='o', linestyle='-')
    plt.xlabel('Bitrate (bpp)')
    plt.ylabel('Average PSNR (dB)')
    plt.title('Average Rate-Distortion Curve Over All Images')
    plt.grid(True)

    rd_plot_path = os.path.join(output_folder, 'average_rdplot.png')
    plt.savefig(rd_plot_path)
    plt.close()
    print("Saved average Rate-Distortion plot:", rd_plot_path)

if __name__ == '__main__':
    main()

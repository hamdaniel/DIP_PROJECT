import os
import pandas as pd
import matplotlib.pyplot as plt

# Output directory for plots
output_dir = 'rd_plots'  # change this to your desired folder
os.makedirs(output_dir, exist_ok=True)

# Read overall CSV files for summary RD curves
duan_df = pd.read_csv('../Duan/outputs/rd_curve.csv')
islam_df = pd.read_csv('../Islam/my_recons/average_rd_values.csv')

# Read per-image detailed CSV files
duan_per_image_df = pd.read_csv('../Duan/outputs/per_image_results.csv') 
islam_per_image_df = pd.read_csv('../Islam/my_recons/all_images_rd_values.csv')  

# Plot overall RD curves
plt.figure(figsize=(10, 6))
plt.plot(islam_df['bpp'], islam_df['avg_psnr'], marker='s', label='Islam')
plt.plot(duan_df['avg_bpp'], duan_df['avg_psnr'], marker='o', label='Duan')
plt.xlabel('Bits per Pixel (bpp)')
plt.ylabel('PSNR (dB)')
plt.title('Rate-Distortion (RD) Curves')
plt.grid(True, linestyle='--', alpha=0.5)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rd_plot.png'), dpi=300)
plt.close()

# Get list of unique images from Islam detailed CSV
images = islam_per_image_df['image_name'].unique()

# Track images where Islam is better
islam_better_images = []

# Generate per-image RD plots
for img in images:
    plt.figure(figsize=(10, 6))

    # Islam per-image data for current image
    islam_img_data = islam_per_image_df[islam_per_image_df['image_name'] == img]
    plt.plot(islam_img_data['bpp'], islam_img_data['psnr'], marker='s', label='Islam')
    max_islam_psnr = islam_img_data['psnr'].max()

    # Duan per-image data for current image
    duan_img_row = duan_per_image_df[duan_per_image_df['image'] == f'{img}.png']
    max_duan_psnr = float('-inf')
    if not duan_img_row.empty:
        # Extract bpp and psnr values for all lambdas for this image
        bpp_cols = [col for col in duan_per_image_df.columns if '_bpp' in col]
        psnr_cols = [col for col in duan_per_image_df.columns if '_psnr' in col]

        bpp_values = duan_img_row[bpp_cols].values.flatten()
        psnr_values = duan_img_row[psnr_cols].values.flatten()
        max_duan_psnr = max(psnr_values)

        plt.plot(bpp_values, psnr_values, marker='o', label='Duan')

    # Compare maximum PSNR
    if max_islam_psnr > max_duan_psnr:
        islam_better_images.append(img)

    plt.xlabel('Bits per Pixel (bpp)')
    plt.ylabel('PSNR (dB)')
    plt.title(f'RD Curve for Image: {img}')
    plt.grid(True, linestyle='--', alpha=0.5)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'rd_plot_{img}.png'), dpi=300)
    plt.close()

# Print images where Islam was better
print("\nImages where Islam was better (higher max PSNR):")
for img in islam_better_images:
    print(img)

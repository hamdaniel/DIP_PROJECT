import os
import pandas as pd
import matplotlib.pyplot as plt

# Output directory for plots
output_dir = 'rd_plots'
os.makedirs(output_dir, exist_ok=True)

# Read CSV files
duan_df = pd.read_csv('../Duan/ssim_with_jpeg/rd_curve.csv')
islam_df = pd.read_csv('../Islam/BSD500/reconstructions/average_rd_values.csv')

# Read per-image detailed CSV files
duan_per_image_df = pd.read_csv('../Duan/ssim_with_jpeg/per_image_results.csv')
islam_per_image_df = pd.read_csv('../Islam/BSD500/reconstructions/all_images_rd_values.csv')

# Separate JPEG and QRes34 data from Duan's results
duan_jpeg_data = duan_df[duan_df['method'] == 'JPEG']
duan_qres_data = duan_df[duan_df['method'] == 'QRes34M']

# Plot overall RD curves
plt.figure(figsize=(10, 6))
plt.plot(islam_df['bpp'], islam_df['avg_ssim'], marker='s', label='Islam')
plt.plot(duan_qres_data['avg_bpp'], duan_qres_data['avg_ssim'], marker='o', label='Duan QRes34')
plt.plot(duan_jpeg_data['avg_bpp'], duan_jpeg_data['avg_ssim'], marker='^', label='JPEG', color='red')

plt.xlabel('Bits per Pixel (bpp)')
plt.ylabel('ssim (dB)')
plt.title('Rate-Distortion (RD) Curves')
plt.grid(True, linestyle='--', alpha=0.5)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ssim_rd_plot.png'), dpi=300)
plt.close()

# Get list of unique images from Islam detailed CSV
images = islam_per_image_df['image_name'].unique()

# Track images where Islam is better
islam_better_images = []

# Generate per-image RD plots
for img in images:
    plt.figure(figsize=(10, 6))

    # Islam per-image data
    islam_img_data = islam_per_image_df[islam_per_image_df['image_name'] == img]
    plt.plot(islam_img_data['bpp'], islam_img_data['ssim'], marker='s', label='Islam')
    max_islam_ssim = islam_img_data['ssim'].max()

    # Duan per-image data
    duan_img_row = duan_per_image_df[duan_per_image_df['image'] == f'{img}.jpg']
    max_duan_ssim = float('-inf')
    
    if not duan_img_row.empty:
        # QRes34 data
        qres_bpp_cols = [col for col in duan_per_image_df.columns if 'QRes34_λ' in col and '_bpp' in col]
        qres_ssim_cols = [col for col in duan_per_image_df.columns if 'QRes34_λ' in col and '_ssim' in col]
        
        qres_bpp_values = duan_img_row[qres_bpp_cols].values.flatten()
        qres_ssim_values = duan_img_row[qres_ssim_cols].values.flatten()
        max_duan_ssim = max(qres_ssim_values)
        
        plt.plot(qres_bpp_values, qres_ssim_values, marker='o', label='Duan QRes34')
        
        # JPEG data
        jpeg_bpp_cols = [col for col in duan_per_image_df.columns if 'JPEG_q' in col and '_bpp' in col]
        jpeg_ssim_cols = [col for col in duan_per_image_df.columns if 'JPEG_q' in col and '_ssim' in col]
        
        jpeg_bpp_values = duan_img_row[jpeg_bpp_cols].values.flatten()
        jpeg_ssim_values = duan_img_row[jpeg_ssim_cols].values.flatten()
        
        plt.plot(jpeg_bpp_values, jpeg_ssim_values, marker='^', label='JPEG', color='red')

    # Compare maximum ssim
    if max_islam_ssim > max_duan_ssim:
        islam_better_images.append(img)

    plt.xlabel('Bits per Pixel (bpp)')
    plt.ylabel('ssim (dB)')
    plt.title(f'RD Curve for Image: {img}')
    plt.grid(True, linestyle='--', alpha=0.5)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'rd_plot_{img}.png'), dpi=300)
    plt.close()

# Print images where Islam was better
print("\nImages where Islam was better (higher max ssim):")
for img in islam_better_images:
    print(img)
def main():
    parser = argparse.ArgumentParser(description='Run QRes34 model progressive coding and visualize results.')
    parser.add_argument('--lmbs', type=int, nargs='+', required=True, help='List of lambda values')
    parser.add_argument('--images', nargs='*', default=None, help='Paths to input images (optional)')
    parser.add_argument('--outdir', type=str, default='./outputs', help='Directory to save outputs')
    args = parser.parse_args()

    allowed_lambdas = {16, 32, 64, 128, 256, 512, 1024, 2048}
    for lmb in args.lmbs:
        if lmb not in allowed_lambdas:
            raise ValueError(f"Invalid lambda: {lmb}. Allowed values are: {sorted(allowed_lambdas)}")

    if args.images is None or len(args.images) == 0:
        # Default to all images in ./images folder
        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
        images = []
        for ext in image_extensions:
            images.extend(glob(os.path.join('images', ext)))
        if not images:
            print("No images found in ./images folder.")
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

    for img_path in images:
        fig_images = []
        fig_titles = []

        # Original image
        original = tvf.to_tensor(Image.open(img_path)).clamp(0,1)
        fig_images.append(original)
        fig_titles.append("Original")

        for lmb in args.lmbs:
            model = models[lmb]
            output, bpp = run_and_return_bpp(model, img_path)
            fig_images.append(output)
            fig_titles.append(f"λ={lmb}\n{bpp:.3f} bpp")

        # Plot and save
        fig, axes = plt.subplots(1, len(fig_images), figsize=(3 * len(fig_images), 3))
        for ax, img, title in zip(axes, fig_images, fig_titles):
            ax.imshow(tvf.to_pil_image(img))
            ax.set_title(title)
            ax.axis('off')

        base_name = os.path.basename(img_path)
        fig_path = os.path.join(args.outdir, f'compare_{base_name}.png')
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved comparison plot to {fig_path}")

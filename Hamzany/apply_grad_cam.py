import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import random
import pandas as pd

from model import CompressionTimePredictor
from dataset import CompressionTimeDatasetFromDF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_image(image_path, image_mean, image_std):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean.tolist(), std=image_std.tolist())
    ])
    return transform(image).unsqueeze(0)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, input_image, iter_num, target_class=None):
        self.model.zero_grad()
        output = self.model(input_image, iter_num)
        if target_class is None:
            target = output
        else:
            target = output[:, target_class]
        target.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * activations, dim=1)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.squeeze().cpu().numpy()
        return cam

def show_cam_on_image(img, mask, alpha=0.5):
    import cv2
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]
    cam = heatmap * alpha + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

if __name__ == "__main__":
    csv_path = '../datasets/BSD500_timings/total_timings_cpu.csv'
    image_dir = '../Islam/BSD500/val_padded'
    model_path = 'best_model.pth'
    output_dir = 'gradcam_outputs'
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    image_mean = torch.tensor([0.485, 0.456, 0.406])
    image_std = torch.tensor([0.229, 0.224, 0.225])

    iternum_mean = df['iter_num'].mean()
    iternum_std = df['iter_num'].std()

    model = CompressionTimePredictor(hidden_size=128, iter_size=16)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    indices = list(range(len(df)))
    random.shuffle(indices)

    grad_cam = GradCAM(model, target_layer=model.conv[3])

    generated_count = 0
    for idx in indices:
        if generated_count >= 2:
            break

        sample_row = df.iloc[idx]
        img_path = os.path.join(image_dir, sample_row['image'])[:-4] + "_padded.png"
        if not os.path.exists(img_path):
            continue
        input_image = preprocess_image(img_path, image_mean, image_std).to(device)

        iter_num_norm = torch.tensor([(sample_row['iter_num'] - iternum_mean) / iternum_std], dtype=torch.float32).to(device).unsqueeze(1)

        with torch.no_grad():
            pred = model(input_image, iter_num_norm).item()

        true_time = sample_row['time']
        abs_diff = abs(pred - true_time)

        if abs_diff > 0.2:
            print('Skipping sample {} due to large difference: {:.4f} s'.format(idx, abs_diff))
            continue

        mask = grad_cam(input_image, iter_num_norm)

        orig_img = Image.open(img_path).convert("RGB")
        orig_img_np = np.array(orig_img).astype(np.float32) / 255
        cam_image = show_cam_on_image(orig_img_np, mask)

        plt.figure(figsize=(5, 5))
        plt.imshow(cam_image)
        plt.axis('off')
        title_str = 'Iterations: {}\nPredicted time: {:.4f}\nGround truth time: {:.4f}'.format(
            sample_row["iter_num"], pred, true_time)
        plt.title(title_str)

        image_name = os.path.splitext(os.path.basename(sample_row['image']))[0]
        iter_num = sample_row['iter_num']
        save_filename = 'gradcam_sample_{}_{}.png'.format(image_name, iter_num)
        save_path = os.path.join(output_dir, save_filename)
        plt.savefig(save_path)
        plt.close()

        print('Saved Grad-CAM image for sample {} to {}'.format(idx, save_path))
        generated_count += 1

    grad_cam.remove_hooks()


from tqdm import tqdm
import torch
import pandas as pd
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from model import CompressionTimePredictor
from dataset import CompressionTimeDatasetFromDF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time

# --- Configuration ---
model_path = 'best_model.pth'
csv_path = '../datasets/coco_patches/dummy_data.csv'
image_dir = '../datasets/coco_patches'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_normalization_values(df, image_dir):
    sum_rgb = torch.zeros(3)
    sumsq_rgb = torch.zeros(3)
    total_pixels = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Computing normalization"):
        img_path = os.path.join(image_dir, row['image'])
        image = Image.open(img_path).convert("RGB")
        image_tensor = transforms.ToTensor()(image)  # shape: [3, H, W]
        
        n_pixels = image_tensor.shape[1] * image_tensor.shape[2]
        sum_rgb += image_tensor.view(3, -1).sum(dim=1)
        sumsq_rgb += (image_tensor.view(3, -1) ** 2).sum(dim=1)
        total_pixels += n_pixels

    image_mean = sum_rgb / total_pixels
    image_std = (sumsq_rgb / total_pixels - image_mean ** 2).sqrt()

    # dummy iternum stats since your dummy CSV is all 0s
    iternum_mean = df.iloc[:, 1:].values.mean()
    iternum_std = df.iloc[:, 1:].values.std()

    print("compute_normalization_values: image_mean={}, image_std={}, iternum_mean={}, iternum_std={}".format(
        image_mean, image_std, iternum_mean, iternum_std))
    
    return image_mean, image_std, iternum_mean, iternum_std

# --- Load model ---
model = CompressionTimePredictor(hidden_size=128, iter_size=16)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# --- Load CSV ---
df = pd.read_csv(csv_path)
image_mean, image_std, iternum_mean, iternum_std = compute_normalization_values(df, image_dir)
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean.tolist(), std=image_std.tolist())
    ])
# --- Prepare transforms and dataset ---
dataset = CompressionTimeDatasetFromDF(df, image_dir, transform=transform)


start_time = time.time()

for i in tqdm(range(len(dataset))):
    (image_tensor, iter_tensor), true_time_tensor = dataset[i]
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
    iter_tensor = iter_tensor.to(device)
    true_time = true_time_tensor.item()

    with torch.no_grad():
        iter_tensor = iter_tensor.unsqueeze(0).to(device)  # Add batch dimension
        pred_time = model(image_tensor, iter_tensor).item()

end_time = time.time()
elapsed = end_time - start_time
print("Model inference over {} samples took {} seconds".format(len(dataset), elapsed))
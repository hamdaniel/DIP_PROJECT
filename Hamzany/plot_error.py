
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

# --- Configuration ---
model_path = 'best_model.pth'
csv_path = '../datasets/BSD500_timings/total_timings_cpu.csv'
image_dir = '../datasets/BSD500_padded'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_normalization_values(df, image_dir):
    pixel_values = []
    for i, row in df.iterrows():
        img_path = os.path.join(image_dir, row['image'])
        image = Image.open(img_path).convert("RGB")
        image_tensor = transforms.ToTensor()(image)
        pixel_values.append(image_tensor.view(3, -1))
    all_pixels = torch.cat(pixel_values, dim=1)
    image_mean = all_pixels.mean(dim=1)
    image_std = all_pixels.std(dim=1)

    iternum_mean = df['iter_num'].mean()
    iternum_std = df['iter_num'].std()

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

# --- Predict and compute errors ---
results = []

for i in tqdm(range(len(dataset))):
    (image_tensor, iter_tensor), true_time_tensor = dataset[i]
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
    iter_tensor = iter_tensor.to(device)
    true_time = true_time_tensor.item()

    with torch.no_grad():
        iter_tensor = iter_tensor.unsqueeze(0).to(device)  # Add batch dimension
        pred_time = model(image_tensor, iter_tensor).item()


    error_pct = abs(pred_time - true_time) / true_time * 100

    results.append({
        'image': df.iloc[i]['image'],  # use df to get image name
        'iter_num': df.iloc[i]['iter_num'],
        'true_time': true_time,
        'pred_time': pred_time,
        'error_pct': error_pct
    })

results_df = pd.DataFrame(results)

# --- Naive guess baseline (mean time per image) ---
naive_mean = df['time'].mean()
results_df['naive_error_pct'] = abs(results_df['true_time'] - naive_mean) / results_df['true_time'] * 100

# --- Plotting ---
mean_errors_by_iter = results_df.groupby('iter_num')['error_pct'].mean()
overall_model_error = results_df['error_pct'].mean()
naive_guess_error = results_df['naive_error_pct'].mean()

plt.figure(figsize=(12, 6))
plt.plot(mean_errors_by_iter.index, mean_errors_by_iter.values, marker='o', label='Model Error by Iteration')
plt.axhline(overall_model_error, color='green', linestyle='--', label='Overall Model Avg Error: {:.2f}%'.format(overall_model_error))
plt.axhline(naive_guess_error, color='red', linestyle='--', label='Naive Guess Avg Error: {:.2f}%'.format(naive_guess_error))
plt.xlabel('Iteration Number')
plt.ylabel('Average % Error')
plt.title('Compression Time Prediction Error Analysis')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("error_by_iteration_count")

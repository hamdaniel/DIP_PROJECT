
import os
import random
import torch
import itertools
import numpy as np
import csv
from torch.utils.data import DataLoader, Subset
from model import CompressionTimePredictor
from dataset import CompressionTimeDatasetFromDF
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import pandas as pd
import math
from tqdm import tqdm
log_path = 'hyperparam_tuning_log.csv'
log_df = pd.read_csv(log_path)
tune_logs = log_df[log_df['phase'] == 'tune']
tune_logs = tune_logs.dropna(subset=['val_loss'])

if not tune_logs.empty:
	tune_logs['val_loss'] = tune_logs['val_loss'].astype(float)
	min_row = tune_logs.loc[tune_logs['val_loss'].idxmin()]
	best_params = (
		float(min_row['lr']),
		int(min_row['batch_size']),
		int(min_row['hidden_size']),
		int(min_row['iter_fc']),
		float(min_row["val_loss"])
	)
	print("Best hyperparameters from log: lr={}, batch_size={}, hidden_size={}, iter_fc={}, val los is: {}".format(*best_params))

def compute_average_percent_error(csv_path, reference_error=0.2):
    df = pd.read_csv(csv_path)

    # Ensure 'time' column exists
    if 'time' not in df.columns:
        raise ValueError("CSV must contain a 'time' column")

    # Calculate percent error: (reference_error / actual_time) * 100
    df['percent_error'] = (reference_error / df['time']) * 100

    avg_percent_error = df['percent_error'].mean()
    print("Average percent that", reference_error, "seconds is, relative to actual times: {:.2f}%".format(avg_percent_error))


def compute_naive_model_percent_error(csv_path):
    df = pd.read_csv(csv_path)

    actual_times = df['time'].values
    mean_time = actual_times.mean()

    percent_errors = abs(mean_time - actual_times) / actual_times * 100
    avg_percent_error = percent_errors.mean()

    print("Average compression time in dataset:", mean_time)
    print("Average percent error for naive model (predicts mean):", round(avg_percent_error, 2), "%")
    
# Example usage
csv_path = "../datasets/BSD500_timings/total_timings_cpu.csv"  # <-- Replace with your CSV file path
compute_average_percent_error(csv_path, math.sqrt(float(min_row["val_loss"])))
compute_naive_model_percent_error(csv_path)
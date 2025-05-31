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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def calculate_transforms(train_dataset):
    # Calculate mean and std from train_dataset images (ToTensor needed first)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)

    mean = 0.
    std = 0.
    total_images = 0

    for (images_iters, _) in loader:
        images = images_iters[0]  # images_iters is (image_tensor, iter_tensor)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    print("Calculated mean: {}, std: {}".format(mean, std))

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

    transform_val_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

    return transform_train, transform_val_test


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for (images, iters), targets in dataloader:
        images = images.to(device)
        iters = iters.to(device)
        targets = targets.to(device)

        outputs = model(images, iters)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for (images, iters), targets in dataloader:
            images = images.to(device)
            iters = iters.to(device)
            targets = targets.to(device)

            outputs = model(images, iters)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)

def write_log_header(log_path):
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'lr', 'batch_size', 'hidden_size', 'epoch', 
            'train_loss', 'val_loss', 'phase'
        ])

def append_log(log_path, row):
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def main(csv_path, image_dir):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    full_df = pd.read_csv(csv_path)
    indices = list(range(len(full_df)))
    random.shuffle(indices)

    train_split = int(0.7 * len(indices))
    val_split = int(0.9 * len(indices))

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    train_df = full_df.iloc[train_indices].reset_index(drop=True)
    val_df = full_df.iloc[val_indices].reset_index(drop=True)
    test_df = full_df.iloc[test_indices].reset_index(drop=True)

    # Create temporary dataset to compute normalization (only ToTensor transform)
    temp_train_dataset = CompressionTimeDatasetFromDF(train_df, image_dir, transform=transforms.ToTensor())

    transform_train, transform_val_test = calculate_transforms(temp_train_dataset)

    train_dataset = CompressionTimeDatasetFromDF(train_df, image_dir, transform=transform_train)
    val_dataset = CompressionTimeDatasetFromDF(val_df, image_dir, transform=transform_val_test)
    test_dataset = CompressionTimeDatasetFromDF(test_df, image_dir, transform=transform_val_test)

    # Hyperparameter search space
    # learning_rates = [0.01, 0.001, 0.0005, 0.0001]
    # batch_sizes = [16, 32]
    # hidden_sizes = [64, 128]
    # iter_fc_sizes = [16, 32]
    learning_rates = [0.0001]
    batch_sizes = [16]
    hidden_sizes = [64]
    iter_fc_sizes = [16]
    max_epochs = 2
    patience = 5  # early stopping patience

    log_path = 'hyperparam_tuning_log.csv'
    write_log_header(log_path)

    best_val_loss = float('inf')
    best_params = None

    for lr, batch_size, hidden_size, iter_fc_size in itertools.product(learning_rates, batch_sizes, hidden_sizes, iter_fc_sizes):
        print("Trying lr={}, batch_size={}, hidden_size={}, iter_fc_size={}".format(lr, batch_size, hidden_size, iter_fc_size))

        model = CompressionTimePredictor(hidden_size=hidden_size, iter_size=iter_fc_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        best_combo_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(1, max_epochs + 1):
            train_loss = train(model, train_loader, criterion, optimizer, device)
            val_loss = evaluate(model, val_loader, criterion, device)

            print("Epoch {} - Train Loss: {:.6f}, Val Loss: {:.6f}".format(epoch, train_loss, val_loss))

            append_log(log_path, [lr, batch_size, hidden_size, epoch, train_loss, val_loss, 'tune'])

            if val_loss < best_combo_val_loss:
                best_combo_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print("Early stopping at epoch {}".format(epoch))
                break

        if best_combo_val_loss < best_val_loss:
            best_val_loss = best_combo_val_loss
            best_params = (lr, batch_size, hidden_size)

    print("Best hyperparameters: lr={}, batch_size={}, hidden_size={}".format(*best_params))

    # Retrain on train + val sets combined from scratch (no loading weights)
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)
    full_train_dataset = CompressionTimeDatasetFromDF(full_train_df, image_dir, transform=transform_train)
    test_dataset = CompressionTimeDatasetFromDF(test_df, image_dir, transform=transform_val_test)

    full_train_loader = DataLoader(full_train_dataset, batch_size=best_params[1], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=best_params[1])

    final_model = CompressionTimePredictor(hidden_size=best_params[2]).to(device)
    optimizer = optim.Adam(final_model.parameters(), lr=best_params[0])
    criterion = nn.MSELoss()

    epochs_no_improve = 0
    best_final_train_loss = float('inf')

    for epoch in range(1, max_epochs + 1):
        train_loss = train(final_model, full_train_loader, criterion, optimizer, device)
        print("Retrain Epoch {} - Train Loss: {:.6f}".format(epoch, train_loss))

        append_log(log_path, [best_params[0], best_params[1], best_params[2], epoch, train_loss, '', 'final_train'])

        if train_loss < best_final_train_loss:
            best_final_train_loss = train_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping final training at epoch {}".format(epoch))
            break

    test_loss = evaluate(final_model, test_loader, criterion, device)
    print("Final Test Loss: {:.6f}".format(test_loss))

    append_log(log_path, [best_params[0], best_params[1], best_params[2], '', '', test_loss, 'final_test'])

    # Save the final model weights
    model_save_path = "best_model.pth"
    torch.save(final_model.state_dict(), model_save_path)
    print("Final model weights saved to", model_save_path)

if __name__ == '__main__':
    csv_path = '../datasets/BSD500_timings/total_timings_cpu.csv'           # <- change to your CSV file path
    image_dir = '../datasets/BSD500_padded'  # <- change to your image folder path
    main(csv_path, image_dir)

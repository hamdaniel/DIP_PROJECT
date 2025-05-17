import subprocess
import torch

print("GPU successfully acquired. " if torch.cuda.is_available() else "Failed to acquire GPU!")
import torch
from network import EncoderCell, DecoderCell, Binarizer

models = [EncoderCell(), DecoderCell(), Binarizer()]

total_params = 0
for model in models:
    total_params += sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total trainable parameters: {}".format(total_params))

import torch
from lvae import get_model

model = get_model('qres34m', lmb=16, pretrained=True)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total trainable parameters: {}".format(total_params))

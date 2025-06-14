import argparse
import os
import time
import csv

import numpy as np
from scipy.misc import imread
import torch
from torch.autograd import Variable

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', required=True, type=str, help='path to encoder model')
parser.add_argument('--input', '-i', required=True, type=str, help='input image')
parser.add_argument('--output', '-o', required=True, type=str, help='output codes')
parser.add_argument('--cuda', '-g', action='store_true', help='enables cuda')
parser.add_argument('--iterations', type=int, default=16, help='unroll iterations')
args = parser.parse_args()

# Read and preprocess image
image = imread(args.input, mode='RGB')
image = torch.from_numpy(
    np.expand_dims(
        np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)), 0))
batch_size, input_channels, height, width = image.size()
assert height % 32 == 0 and width % 32 == 0

with torch.no_grad():
    image = Variable(image)

# Load model
import network

encoder = network.EncoderCell()
binarizer = network.Binarizer()
decoder = network.DecoderCell()

encoder.eval()
binarizer.eval()
decoder.eval()

device = torch.device("cuda" if args.cuda else "cpu")
encoder.load_state_dict(torch.load(args.model, map_location=device))
binarizer.load_state_dict(
    torch.load(args.model.replace('encoder', 'binarizer'), map_location=device))
decoder.load_state_dict(torch.load(args.model.replace('encoder', 'decoder'), map_location=device))

with torch.no_grad():
    encoder_h_1 = (torch.zeros(batch_size, 256, height // 4, width // 4),
                   torch.zeros(batch_size, 256, height // 4, width // 4))
    encoder_h_2 = (torch.zeros(batch_size, 512, height // 8, width // 8),
                   torch.zeros(batch_size, 512, height // 8, width // 8))
    encoder_h_3 = (torch.zeros(batch_size, 512, height // 16, width // 16),
                   torch.zeros(batch_size, 512, height // 16, width // 16))

    decoder_h_1 = (torch.zeros(batch_size, 512, height // 16, width // 16),
                   torch.zeros(batch_size, 512, height // 16, width // 16))
    decoder_h_2 = (torch.zeros(batch_size, 512, height // 8, width // 8),
                   torch.zeros(batch_size, 512, height // 8, width // 8))
    decoder_h_3 = (torch.zeros(batch_size, 256, height // 4, width // 4),
                   torch.zeros(batch_size, 256, height // 4, width // 4))
    decoder_h_4 = (torch.zeros(batch_size, 128, height // 2, width // 2),
                   torch.zeros(batch_size, 128, height // 2, width // 2))

if args.cuda:
    encoder = encoder.cuda()
    binarizer = binarizer.cuda()
    decoder = decoder.cuda()

    image = image.cuda()
    encoder_h_1 = (encoder_h_1[0].cuda(), encoder_h_1[1].cuda())
    encoder_h_2 = (encoder_h_2[0].cuda(), encoder_h_2[1].cuda())
    encoder_h_3 = (encoder_h_3[0].cuda(), encoder_h_3[1].cuda())
    decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
    decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
    decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
    decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())

# Initialize residual and code list
codes = []
res = image - 0.5

# Timing setup
image_name = os.path.basename(args.input)
timing_file = 'timings_gpu.csv'
iteration_times = []
cumulative_time = 0.0

# Inference loop
for iters in range(args.iterations):
    start_time = time.time()

    with torch.no_grad():
        encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
            res, encoder_h_1, encoder_h_2, encoder_h_3)

        code = binarizer(encoded)

        output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
            code, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

        res = res - output
        codes.append(code.data.cpu().numpy())

    elapsed = time.time() - start_time
    cumulative_time += elapsed
    iteration_times.append(cumulative_time)

    print("Iter: {:02d}; Loss: {:.06f}; Cumulative Time: {:.4f}s".format(
        iters, res.data.abs().mean(), cumulative_time))

# Convert and save codes
codes = (np.stack(codes).astype(np.int8) + 1) // 2
export = np.packbits(codes.reshape(-1))
np.savez_compressed(args.output, shape=codes.shape, codes=export)

# Save timing to collective CSV
header = ['image'] + ['iter_{}'.format(i + 1) for i in range(args.iterations)]
write_header = not os.path.exists(timing_file)

with open(timing_file, 'a', newline='') as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(header)
    writer.writerow([image_name] + iteration_times)

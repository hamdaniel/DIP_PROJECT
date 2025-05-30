import os
import argparse
import time
import csv

import numpy as np
from scipy.misc import imread, imresize, imsave

import torch
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str, help='path to model')
parser.add_argument('--input', required=True, type=str, help='input codes')
parser.add_argument('--output', default='.', type=str, help='output folder')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--iterations', type=int, default=16, help='unroll iterations')
args = parser.parse_args()

content = np.load(args.input)
codes = np.unpackbits(content['codes'])
codes = np.reshape(codes, content['shape']).astype(np.float32) * 2 - 1

codes = torch.from_numpy(codes)
iters, batch_size, channels, height, width = codes.size()
height = height * 16
width = width * 16

with torch.no_grad():
    codes = Variable(codes)

import network

decoder = network.DecoderCell()
decoder.eval()

device = torch.device("cuda" if args.cuda else "cpu")
decoder.load_state_dict(torch.load(args.model, map_location=device))

with torch.no_grad():
    decoder_h_1 = (torch.zeros(batch_size, 512, height // 16, width // 16),
                   torch.zeros(batch_size, 512, height // 16, width // 16))
    decoder_h_2 = (torch.zeros(batch_size, 512, height // 8, width // 8),
                   torch.zeros(batch_size, 512, height // 8, width // 8))
    decoder_h_3 = (torch.zeros(batch_size, 256, height // 4, width // 4),
                   torch.zeros(batch_size, 256, height // 4, width // 4))
    decoder_h_4 = (torch.zeros(batch_size, 128, height // 2, width // 2),
                   torch.zeros(batch_size, 128, height // 2, width // 2))

if args.cuda:
    decoder = decoder.cuda()
    codes = codes.cuda()
    decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
    decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
    decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
    decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())

image = torch.zeros(1, 3, height, width) + 0.5
if args.cuda:
    image = image.cuda()

# Timing setup
image_name = os.path.basename(args.input)
timing_file = 'decode_timings_gpu.csv'
iteration_times = []
cumulative_time = 0.0

for i in range(min(args.iterations, codes.size(0))):
    start_time = time.time()

    with torch.no_grad():
        output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
            codes[i], decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
        image = image + output.data

    elapsed = time.time() - start_time
    cumulative_time += elapsed
    iteration_times.append(cumulative_time)

    imsave(
        os.path.join(args.output, '{:02d}.png'.format(i)),
        np.squeeze(image.cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
        .transpose(1, 2, 0))

    print("Iter {:02d} - Cumulative Decode Time: {:.4f}s".format(i, cumulative_time))

# Save decode timings
header = ['image'] + ['iter_{}'.format(i + 1) for i in range(len(iteration_times))]
write_header = not os.path.exists(timing_file)

with open(timing_file, 'a', newline='') as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(header)
    writer.writerow([image_name] + iteration_times)

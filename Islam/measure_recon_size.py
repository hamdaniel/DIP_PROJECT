import os
import math
import time
import csv
import glob
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import network  # Your encoder/decoder models here
import warnings
warnings.filterwarnings("ignore")

CODES_DIR = "BSD500/codes_cpu_tmp"
DECODED_DIR = "BSD500/decoded_cpu_tmp"

if not os.path.exists(CODES_DIR):
    os.makedirs(CODES_DIR)
if not os.path.exists(DECODED_DIR):
    os.makedirs(DECODED_DIR)

# Model checkpoints (change paths if needed)
ENCODER_MODEL_PATH = "checkpoint/encoder_epoch_00000001.pth"
DECODER_MODEL_PATH = "checkpoint/decoder_epoch_00000001.pth"

# Number of iterations for encode/decode
ITERATIONS = 16

device = torch.device("cpu")  # Change to 'cuda' if you want GPU

# Load encoder model
encoder = network.EncoderCell()
encoder.load_state_dict(torch.load(ENCODER_MODEL_PATH, map_location=device))
encoder.to(device)
encoder.eval()

# Load binarizer model (from your encoder.py)
binarizer = network.Binarizer()
binarizer.to(device)
binarizer.eval()

# Load decoder model
decoder = network.DecoderCell()
decoder.load_state_dict(torch.load(DECODER_MODEL_PATH, map_location=device))
decoder.to(device)
decoder.eval()


def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = torch.FloatTensor(np.array(img).transpose(2, 0, 1) / 255.0).unsqueeze(0)
    return img_tensor


def encode_image(img_tensor, iterations=ITERATIONS):
    img_tensor = img_tensor.to(device)
    batch_size, channels, height, width = img_tensor.size()

    with torch.no_grad():
        res = img_tensor - 0.5
        encoder_h_1 = (torch.zeros(batch_size, 256, height // 4, width // 4, device=device),
                       torch.zeros(batch_size, 256, height // 4, width // 4, device=device))
        encoder_h_2 = (torch.zeros(batch_size, 512, height // 8, width // 8, device=device),
                       torch.zeros(batch_size, 512, height // 8, width // 8, device=device))
        encoder_h_3 = (torch.zeros(batch_size, 512, height // 16, width // 16, device=device),
                       torch.zeros(batch_size, 512, height // 16, width // 16, device=device))

        codes_list = []
        for _ in range(iterations):
            encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(res, encoder_h_1, encoder_h_2, encoder_h_3)
            code = binarizer(encoded)
            codes_list.append(code.data.cpu().numpy())

        codes_np = (np.stack(codes_list).astype(np.int8) + 1) // 2
        packed_codes = np.packbits(codes_np.reshape(-1))

        shape = codes_np.shape
    return packed_codes, shape


def save_codes(packed_codes, shape, filepath):
    np.savez_compressed(filepath, codes=packed_codes, shape=shape)


def decode_codes(codes_path, output_dir, iterations=ITERATIONS):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    content = np.load(codes_path)
    codes = np.unpackbits(content['codes'])
    codes = np.reshape(codes, content['shape']).astype(np.float32) * 2 - 1
    codes = torch.from_numpy(codes).to(device)

    iters, batch_size, channels, height, width = codes.size()
    height = height * 16
    width = width * 16

    with torch.no_grad():
        decoder_h_1 = (torch.zeros(batch_size, 512, height // 16, width // 16, device=device),
                       torch.zeros(batch_size, 512, height // 16, width // 16, device=device))
        decoder_h_2 = (torch.zeros(batch_size, 512, height // 8, width // 8, device=device),
                       torch.zeros(batch_size, 512, height // 8, width // 8, device=device))
        decoder_h_3 = (torch.zeros(batch_size, 256, height // 4, width // 4, device=device),
                       torch.zeros(batch_size, 256, height // 4, width // 4, device=device))
        decoder_h_4 = (torch.zeros(batch_size, 128, height // 2, width // 2, device=device),
                       torch.zeros(batch_size, 128, height // 2, width // 2, device=device))

        image = torch.zeros(batch_size, 3, height, width, device=device) + 0.5

        iteration_times = []
        cumulative_time = 0.0

        for i in range(min(iterations, codes.size(0))):
            start_time = time.time()
            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                codes[i], decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
            image = image + output
            elapsed = time.time() - start_time
            cumulative_time += elapsed
            iteration_times.append(cumulative_time)

            print("\r    Decode Iteration {}/{} - Cumulative Time: {:.4f}s".format(i + 1, iterations, cumulative_time), end='')

            im_np = image.squeeze().cpu().numpy().clip(0, 1).transpose(1, 2, 0) * 255
            im_pil = Image.fromarray(im_np.astype(np.uint8))
            im_pil.save(os.path.join(output_dir, '{:02d}.png'.format(i)))

    return iteration_times


def main():
    
    for idx, img_path in enumerate(["BSD500/val_padded/102061_padded.png"], 1):
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        
        img_tensor = load_image(img_path)
        start_enc = time.time()
        packed_codes, shape = encode_image(img_tensor, iterations=ITERATIONS)         
        codes_path = os.path.join(CODES_DIR, "{}.npz".format(image_name))
        save_codes(packed_codes, shape, codes_path)        
        decode_iter_times = decode_codes(codes_path, os.path.join(DECODED_DIR, image_name), iterations=ITERATIONS)

if __name__ == "__main__":
    main()

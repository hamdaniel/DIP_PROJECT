import csv

encode_file = 'encode_timings_cpu.csv'
decode_file = 'decode_timings_cpu.csv'
output_file = 'timings_cpu.csv'

# Load CSV into dictionary: { image_name: [times...] }
def load_csv(path):
    data = {}
    with open(path, newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            data[row[0]] = list(map(float, row[1:]))
    return headers[1:], data

encode_iters, encode_data = load_csv(encode_file)
decode_iters, decode_data = load_csv(decode_file)

assert encode_iters == decode_iters, "Mismatch in iteration columns"

# Merge
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image'] + ['iter_{}_total'.format(i+1) for i in range(len(encode_iters))])
    for image in encode_data:
        if image in decode_data:
            total_times = [e + d for e, d in zip(encode_data[image], decode_data[image])]
            writer.writerow([image] + total_times)
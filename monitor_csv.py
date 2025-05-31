import time
import csv

csv_path = "Hamzany/hyperparam_tuning_log.csv"
default_iter_fc = 16
check_interval = 5  # seconds

prev_hparams = None  # For logging
prev_epoch = None    # For model boundary detection

def correct_csv(csv_path, last_line_count):
    global prev_hparams, prev_epoch

    with open(csv_path, "r") as f:
        lines = f.readlines()

    header = lines[0].strip().split(",")
    corrected_lines = [",".join(header) + "\n"]
    prev_iter_fc = None

    for i in range(1, len(lines)):
        row = lines[i].strip().split(",")

        if len(row) == 7:
            row.insert(3, "")  # Add placeholder for iter_fc
            try:
                current_epoch = int(row[4])
            except ValueError:
                current_epoch = None

            is_new_model = current_epoch == 1 and prev_epoch not in [None, 2]

            if is_new_model:
                row[3] = str(default_iter_fc)
            else:
                row[3] = str(prev_iter_fc if prev_iter_fc is not None else default_iter_fc)

            prev_iter_fc = int(row[3])
            prev_epoch = current_epoch
            corrected_lines.append(",".join(row) + "\n")

        elif len(row) == 8:
            corrected_lines.append(",".join(row) + "\n")
            try:
                prev_iter_fc = int(row[3])
                prev_epoch = int(row[4])
            except ValueError:
                prev_iter_fc = None
                prev_epoch = None
        else:
            continue

    # Rewrite corrected file
    with open(csv_path, "w") as f:
        f.writelines(corrected_lines)

    # Now reload corrected file and only print new lines
    with open(csv_path, "r") as f:
        corrected_lines = f.readlines()

    new_lines = corrected_lines[last_line_count:]
    updated_line_count = len(corrected_lines)

    for line in new_lines:
        parts = line.strip().split(",")
        if len(parts) != 8:
            continue
        lr, batch_size, hidden_size, iter_fc, epoch, train_loss, val_loss, phase = parts
        current_hparams = (lr, batch_size, hidden_size, iter_fc)

        if prev_hparams != current_hparams:
            print(f"\n--- New model started ---")
            print(f"Hyperparameters: lr={lr}, batch_size={batch_size}, hidden_size={hidden_size}, iter_fc={iter_fc}")
            prev_hparams = current_hparams

        print(f"on epoch: {epoch.rjust(2)} on the {phase} phase, the training loss was: {float(train_loss):.4f} and the validation loss was: {float(val_loss):.4f}")

    return updated_line_count

if __name__ == "__main__":
    print("Watching CSV file for updates...\n")
    last_line_count = 1  # Skip header

    while True:
        try:
            with open(csv_path, "r") as f:
                current_lines = f.readlines()

            if len(current_lines) > last_line_count:
                last_line_count = correct_csv(csv_path, last_line_count)

        except Exception as e:
            print(f"Error: {e}")

        time.sleep(check_interval)

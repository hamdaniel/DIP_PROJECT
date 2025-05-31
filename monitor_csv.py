import time
import csv

csv_path = "Hamzany/hyperparam_tuning_log.csv"
possible_iter_fc = [16, 32]
check_interval = 5  # seconds

prev_hparams = None  # Global variable to remember previous hyperparameters

def infer_iter_fc(prev_iter_fc):
    return [val for val in possible_iter_fc if val != prev_iter_fc][0]

def correct_csv(csv_path, last_line_count):
    global prev_hparams  # Allow persistent tracking across calls

    with open(csv_path, "r") as f:
        lines = f.readlines()

    header = lines[0].strip().split(",")
    corrected_lines = [lines[0]]
    prev_iter_fc = None
    prev_epoch = None

    new_lines = lines[last_line_count:]
    updated_line_count = len(lines)

    for i in range(1, len(lines)):
        row = lines[i].strip().split(",")

        if len(row) == 7:
            row.insert(3, "")  # Empty iter_fc

            try:
                current_epoch = int(row[4])
            except ValueError:
                current_epoch = None

            if prev_iter_fc is None:
                row[3] = str(possible_iter_fc[0])
            else:
                if current_epoch == 0 and prev_epoch is not None and prev_epoch != 0:
                    row[3] = str(infer_iter_fc(prev_iter_fc))
                else:
                    row[3] = str(prev_iter_fc)

            prev_iter_fc = int(row[3])
            prev_epoch = current_epoch
            corrected_lines.append(",".join(row) + "\n")
        else:
            corrected_lines.append(lines[i])
            try:
                prev_iter_fc = int(row[3])
                prev_epoch = int(row[4])
            except ValueError:
                prev_iter_fc = None
                prev_epoch = None

    # Write back corrected file
    with open(csv_path, "w") as f:
        f.writelines(corrected_lines)

    # Log meaningful output for the new lines
    for line in new_lines:
        parts = line.strip().split(",")
        if len(parts) != 8:
            continue  # malformed line (still being written?)
        lr, batch_size, hidden_size, iter_fc, epoch, train_loss, val_loss, phase = parts

        current_hparams = (lr, batch_size, hidden_size, iter_fc)
        if prev_hparams != current_hparams:
            print(f"\n--- New model started ---")
            print(f"Hyperparameters: lr={lr}, batch_size={batch_size}, hidden_size={hidden_size}, iter_fc={iter_fc}")
            prev_hparams = current_hparams

        # Pad epoch to 2-character width
        epoch_display = epoch.rjust(2)

        train_loss_fmt = f"{float(train_loss):.4f}"
        val_loss_fmt = f"{float(val_loss):.4f}"

        print(f"on epoch: {epoch_display} on the {phase} phase, the training loss was: {train_loss_fmt} and the validation loss was: {val_loss_fmt}")

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

import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv('hyperparam_tuning_log.csv')  # Replace with your actual file path

# Split by phase
tune_df = df[df['phase'] == 'tune']
final_train_df = df[df['phase'] == 'final_train']
final_test_df = df[df['phase'] == 'final_test']

# Get hyperparameters used in final training
final_hparams = final_train_df.iloc[0][['lr', 'batch_size', 'hidden_size', 'iter_fc']].to_dict()

# Filter tuning logs that match final training hyperparams
matching_tune_df = tune_df[
    (tune_df['lr'] == final_hparams['lr']) &
    (tune_df['batch_size'] == final_hparams['batch_size']) &
    (tune_df['hidden_size'] == final_hparams['hidden_size']) &
    (tune_df['iter_fc'] == final_hparams['iter_fc'])
]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot final training loss
plt.plot(final_train_df['epoch'], final_train_df['train_loss'], label='Final Training Loss', color='blue', marker='o')

# Plot validation loss from tuning
plt.plot(matching_tune_df['epoch'], matching_tune_df['val_loss'], label='Validation Loss', color='orange', marker='x')

# Plot horizontal line for final test loss
if not final_test_df.empty:
    test_loss = final_test_df['val_loss'].values[0]  # or train_loss depending on where it's stored
    plt.axhline(test_loss, color='green', linestyle='--', label='Test Loss = %.4f' % test_loss)

# Add labels, legend, and grid
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training, Validation, and Test Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot to a file
plt.savefig('loss_plot.png')

# Optional: close the plot if running in a script
plt.close()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load CSV
    csv_path = 'timings_cpu.csv'  # <- Change to your actual CSV file path
    df = pd.read_csv(csv_path)

    # Extract iteration columns
    iter_cols = [col for col in df.columns if col.startswith('iter_')]
    iterations = list(range(1, len(iter_cols) + 1))
    iter_data = df[iter_cols]

    # Identify if any single row holds the max value for every iteration
    max_per_col = iter_data.max()
    row_is_max_everywhere = (iter_data == max_per_col).all(axis=1)

    # Optionally remove the row that is max everywhere
    # if row_is_max_everywhere.sum() == 1:
    #     iter_data = iter_data[~row_is_max_everywhere]

    # Compute total compression time per image (sum of iterations)
    total_times = iter_data.sum(axis=1)

    # Get top 3 and bottom 1 image indices
    top3_indices = total_times.nlargest(3).index
    lowest_index = total_times.idxmin()

    # Print image identifiers (assumes df has an identifying column, else use index)
    print("Top 3 images with highest total compression times:")
    for i in top3_indices:
        if 'filename' in df.columns:
            identifier = df.loc[i, 'filename']
        else:
            identifier = i
        print("  - {}: {:.4f} s".format(identifier, total_times[i]))

    print("\nImage with the lowest total compression time:")
    if 'filename' in df.columns:
        identifier = df.loc[lowest_index, 'filename']
    else:
        identifier = lowest_index
    print("  - {}: {:.4f} s".format(identifier, total_times[lowest_index]))

    # --- Plot 1: Scatter plot (all data points) ---
    plt.figure(figsize=(12, 6))
    for _, row in iter_data.iterrows():
        plt.scatter(iterations, row, alpha=0.5, s=10)
    plt.xlabel('Iteration Number')
    plt.ylabel('Time [s]')
    plt.title('Scatter Plot of Iteration Times')
    plt.grid(True)
    plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 2: Line plot (min, max, mean, range) ---
    max_times = iter_data.max(axis=0)
    min_times = iter_data.min(axis=0)
    mean_times = iter_data.mean(axis=0)
    range_times = max_times - min_times

    plt.figure(figsize=(12, 6))
    plt.plot(iterations, max_times, label='Max Time', color='red')
    plt.plot(iterations, min_times, label='Min Time', color='green')
    plt.plot(iterations, mean_times, label='Average Time', color='blue')
    plt.plot(iterations, range_times, label='Range (Max - Min)', color='orange', linestyle='--')
    plt.xlabel('Iteration Number')
    plt.ylabel('Time [s]')
    plt.title('Min, Max, Average, and Range of Iteration Times')
    plt.legend()
    plt.grid(True)
    plt.savefig('min_max_avg_range_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 3: Variance plot ---
    var_times = iter_data.var(axis=0)

    plt.figure(figsize=(12, 6))
    plt.plot(iterations, var_times, label='Variance', color='purple')
    plt.xlabel('Iteration Number')
    plt.ylabel('Variance [s²]')
    plt.title('Variance in Iteration Times')
    plt.grid(True)
    plt.savefig('variance_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()

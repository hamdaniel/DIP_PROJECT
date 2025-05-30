import pandas as pd
import matplotlib.pyplot as plt

def load_iter_csv(path):
    df = pd.read_csv(path)
    iter_cols = [col for col in df.columns if col.startswith('iter_')]
    return df[iter_cols].mean(axis=0)

def load_latent_csv(path):
    df = pd.read_csv(path)
    latent_cols = [col for col in df.columns if col.startswith('time_for_')]
    return df[latent_cols].mean(axis=0)

def main():
    iter_csv_1 = 'timings_cleaned.csv'
    iter_csv_2 = 'timings_gpu_cleaned.csv'
    latent_csv_1 = '../Duan/timings_cpu.csv'
    latent_csv_2 = '../Duan/timings_gpu.csv'

    avg_iter_1 = load_iter_csv(iter_csv_1)
    avg_iter_2 = load_iter_csv(iter_csv_2)
    avg_latent_1 = load_latent_csv(latent_csv_1)
    avg_latent_2 = load_latent_csv(latent_csv_2)

    plt.figure(figsize=(12, 6))

    # Plot Islam iter CSV curves (CPU and GPU)
    plt.plot(range(1, len(avg_iter_1) + 1), avg_iter_1, label='Islam (CPU)')
    plt.plot(range(1, len(avg_iter_2) + 1), avg_iter_2, label='Islam (GPU)')

    # Calculate the average over all latent steps for Duan CPU and GPU
    avg_latent_1_mean = avg_latent_1.mean()
    avg_latent_2_mean = avg_latent_2.mean()

    # Plot horizontal lines for Duan CPU and GPU averages (instead of their curves)
    plt.hlines(avg_latent_1_mean, xmin=1, xmax=len(avg_iter_1), colors='red', linestyles='--', label='Duan (CPU)')
    plt.hlines(avg_latent_2_mean, xmin=1, xmax=len(avg_iter_2), colors='green', linestyles='--', label='Duan (GPU)')

    plt.xlabel('Iteration / Latent Step')
    plt.ylabel('Average time [s]')
    # plt.yscale('log')
    plt.title('Islam and Duan Timings Comparison')
    plt.legend()
    plt.grid(True)

    plt.savefig('timings_comparison.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()

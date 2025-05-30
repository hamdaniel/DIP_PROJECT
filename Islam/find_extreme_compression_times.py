import pandas as pd

def find_extreme_compression_times(csv_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Rename the first column to 'image'
    df.rename(columns={df.columns[0]: 'image'}, inplace=True)

    # Compute total compression time
    df['total_time'] = df.iloc[:, 1:].sum(axis=1)

    # Get top 3 images with highest total time
    top_3 = df.nlargest(3, 'total_time')

    # Get image with the lowest total time
    lowest = df.nsmallest(1, 'total_time')

    print("Top 3 images with highest total compression times:")
    for i in range(len(top_3)):
        print("  -", top_3.iloc[i]['image'])

    print("\nImage with the lowest total compression time:")
    print("  -", lowest.iloc[0]['image'])

# Example usage
csv_file_path = 'timings_cpu.csv'  # Replace this with your actual file path
find_extreme_compression_times(csv_file_path)

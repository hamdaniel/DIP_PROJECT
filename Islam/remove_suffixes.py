import pandas as pd

def clean_csv_suffix(input_path, output_path, suffix_to_remove):
    # Read the CSV
    df = pd.read_csv(input_path)

    # Remove the suffix from the first column
    first_col = df.columns[0]
    df[first_col] = df[first_col].str.replace(suffix_to_remove + r'$', '', regex=True)

    # Save the cleaned CSV
    df.to_csv(output_path, index=False)
    # print(f"Saved cleaned file to: {output_path}")

# Example usage
csv2_path = 'decode_timings_gpu.csv'
csv1_path = 'timings_gpu.csv'

clean_csv_suffix(csv1_path, 'timings_gpu_cleaned.csv', '_padded.png')
clean_csv_suffix(csv2_path, 'decode_timings_gpu_cleaned.csv', '.npz')

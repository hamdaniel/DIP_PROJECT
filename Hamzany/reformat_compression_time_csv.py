import pandas as pd

# File paths
input_path = '../datasets/BSD500_timings/timings_cpu.csv'     # Replace with your actual path
output_path = '../datasets/BSD500_timings/total_timings_cpu.csv'   # Output file path

# Read the original CSV
df = pd.read_csv(input_path)

# Melt the wide-format dataframe into long format
df_melted = pd.melt(df, id_vars=['image'], var_name='iter', value_name='time')

# Extract iteration number from column name like "iter_1" -> 1
df_melted['iter_num'] = df_melted['iter'].str.extract(r'iter_(\d+)', expand=False).astype(int)

# Add .png suffix to image column
df_melted['image'] = df_melted['image'].astype(str) + '.png'

# Reorder columns
df_final = df_melted[['image', 'iter_num', 'time']]

# Write to output CSV
df_final.to_csv(output_path, index=False)

print("Converted CSV saved to:", output_path)

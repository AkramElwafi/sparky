import os
import time
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Function to calculate statistics for a column
def calculate_statistics(col):
    stats = {
        'Mean': df[col].mean(),
        'StdDev': df[col].std(),
        'Min': df[col].min(),
        'Max': df[col].max(),
        '25th_Percentile': df[col].quantile(0.25),
        '75th_Percentile': df[col].quantile(0.75),
        'Kurtosis': df[col].kurtosis(),
        'Skewness': df[col].skew(),
        'Median': df[col].median(),
        'Mode': df[col].mode()[0],
        'Variance': df[col].var(),
        'Range': df[col].max() - df[col].min()
    }
    return col, stats

# Function to perform complex computations for a column
def perform_complex_computations(col):
    for _ in range(10):
        df[f'{col}_rolling_mean'] = df[col].rolling(window=10000).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=10000).std()
        df[f'{col}_expanding_mean'] = df[col].expanding().mean()
        df[f'{col}_expanding_std'] = df[col].expanding().std()

# Function to perform additional complex computations for a column
def perform_additional_complex_computations(col):
    for _ in range(10):
        df[f'{col}_log'] = np.log1p(df[col])
        df[f'{col}_sqrt'] = np.sqrt(df[col])
        df[f'{col}_cumsum'] = df[col].cumsum()
        df[f'{col}_cumprod'] = df[col].cumprod()
        df[f'{col}_exp'] = np.exp(df[col])
        df[f'{col}_sin'] = np.sin(df[col])
        df[f'{col}_cos'] = np.cos(df[col])
        df[f'{col}_tan'] = np.tan(df[col])
        df[f'{col}_diff'] = df[col].diff()
        df[f'{col}_pct_change'] = df[col].pct_change()

start_time = time.time()

input_path = r'sparky/build/workspace/data/generated_2mill_data.csv'
df = pd.read_csv(input_path)
df = df.drop(['ID', 'name'], axis=1)

columns = df.columns

statistics = {
    'Mean': [],
    'StdDev': [],
    'Min': [],
    'Max': [],
    '25th_Percentile': [],
    '75th_Percentile': [],
    'Kurtosis': [],
    'Skewness': [],
    'Median': [],
    'Mode': [],
    'Variance': [],
    'Range': []
}

# Calculate statistics using multithreading
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(calculate_statistics, col) for col in columns]
    for future in tqdm(futures):
        col, stats = future.result()
        for stat, value in stats.items():
            statistics[stat].append(value)

stats_df = pd.DataFrame(statistics, index=columns)
print(stats_df)

print("Starting complex computations...")
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(perform_complex_computations, col) for col in columns]
    for future in tqdm(futures):
        future.result()

print("Starting additional complex computations...")
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(perform_additional_complex_computations, col) for col in columns]
    for future in tqdm(futures):
        future.result()

print("Starting groupby operations...")
for i in tqdm(range(1000)):
    grouped_stats = df.groupby(df.index // 100).agg({
        'Moisture content': ['mean', 'std', 'min', 'max'],
        'Volatile matter': ['mean', 'std', 'min', 'max'],
        'Fixed carbon': ['mean', 'std', 'min', 'max'],
        'Carbon': ['mean', 'std', 'min', 'max'],
        'Hydrogen': ['mean', 'std', 'min', 'max'],
        'Net calorific value (LHV)': ['mean', 'std', 'min', 'max']
    })

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")

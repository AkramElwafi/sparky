import os
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

start_time = time.time()

# Read data into a Pandas DataFrame
output_path = 'docker-sparker/spark-standalone-cluster-on-docker/pd_spark_comparaison/extended_800_filtered_csv.csv'
df = pd.read_csv(output_path)

# Remove 'biomass_id' and 'biomass_name' columns
df = df.drop(['biomass_id', 'biomass_name'], axis=1)

# List of columns to calculate statistics for
columns = df.columns

# Initialize statistics dictionary
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

# Initialize progress bar
for col in tqdm(columns):
    statistics['Mean'].append(df[col].mean())
    statistics['StdDev'].append(df[col].std())
    statistics['Min'].append(df[col].min())
    statistics['Max'].append(df[col].max())
    statistics['25th_Percentile'].append(df[col].quantile(0.25))
    statistics['75th_Percentile'].append(df[col].quantile(0.75))
    statistics['Kurtosis'].append(df[col].kurtosis())
    statistics['Skewness'].append(df[col].skew())
    statistics['Median'].append(df[col].median())
    statistics['Mode'].append(df[col].mode()[0])
    statistics['Variance'].append(df[col].var())
    statistics['Range'].append(df[col].max() - df[col].min())

# Create a DataFrame for the statistics
stats_df = pd.DataFrame(statistics, index=columns)

# Save statistics to a CSV file
stats_df.to_csv('spark-standalone-cluster-on-docker/pd_spark_comparaison/pandas_statistics.csv')

print(stats_df)

# Complex computation: Multiple rolling statistics and groupby operations
print("Starting complex computations...")
for col in tqdm(columns):
    for _ in range(10):  # Increase complexity by repeating operations
        df[f'{col}_rolling_mean'] = df[col].rolling(window=10000).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=10000).std()
        df[f'{col}_expanding_mean'] = df[col].expanding().mean()
        df[f'{col}_expanding_std'] = df[col].expanding().std()

# Adding more complex computations
print("Starting additional complex computations...")
for col in tqdm(columns):
    for _ in range(10):  # Increase complexity by repeating operations
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

# Groupby operation
print("Starting groupby operations...")
for i in tqdm(range(1000)):  # Significantly increase the number of groupby operations
    grouped_stats = df.groupby(df.index // 100).agg({
        'Moisture content': ['mean', 'std', 'min', 'max'],
        'Volatile matter': ['mean', 'std', 'min', 'max'],
        'Fixed carbon': ['mean', 'std', 'min', 'max'],
        'Carbon': ['mean', 'std', 'min', 'max'],
        'Hydrogen': ['mean', 'std', 'min', 'max'],
        'Net calorific value (LHV)': ['mean', 'std', 'min', 'max']
    })

# Save grouped statistics to a CSV file
grouped_stats.to_csv('spark-standalone-cluster-on-docker/pd_spark_comparaison/pandas_grouped_statistics.csv')

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")

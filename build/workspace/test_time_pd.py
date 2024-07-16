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

output_path = 'C:/Users/asus/Desktop/docker-sparker/spark-standalone-cluster-on-docker/build/workspace/data/generated_2millions_data.csv'
df = pd.read_csv(output_path)

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

stats_df = pd.DataFrame(statistics, index=columns)

print(stats_df)

print("Starting complex computations...")
for col in tqdm(columns):
    for _ in range(10):  
        df[f'{col}_rolling_mean'] = df[col].rolling(window=10000).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=10000).std()
        df[f'{col}_expanding_mean'] = df[col].expanding().mean()
        df[f'{col}_expanding_std'] = df[col].expanding().std()

print("Starting additional complex computations...")
for col in tqdm(columns):
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

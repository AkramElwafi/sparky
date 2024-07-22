#code to generate rows (2 millions in this case)
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import random
import string
from tqdm import tqdm

start_time = time.time()

df = pd.read_csv("C:/Users/asus/Desktop/spark/sparkenv/final_filtered_csv.csv")

def remove_useless_rows(data):
    names_to_remove = data[(data["property"] == "Moisture content") & (data["ar_value"] == 0)]["biomass_name"].unique()
    names_to_remove2 = data[data["property"].isin(["C16:0 Palmitic", "IDT (initial deformation temperature)", "Cellulose", "Total ash + biochemical"])]["biomass_name"].unique()
    filtered_data = data[~data["biomass_name"].isin(names_to_remove)]
    filtered_data = filtered_data[~filtered_data["biomass_name"].isin(names_to_remove2)]
    return filtered_data

def to_panda(df):
    output_path = 'C:/Users/asus/Desktop/spark/sparkenv/filteredcsv.csv'
    df.to_csv(output_path, index=False)
    print("Filtered DataFrame has been saved to:", output_path)
    return df

def pivotate_df(df):
    required_properties = ["Moisture content", "Ash content", "Volatile matter", "Fixed carbon", "Carbon", "Hydrogen", "Oxygen", "Nitrogen", "Net calorific value (LHV)"]
    valid_ids = df[df["property"] == "Net calorific value (LHV)"]["biomass_id"].unique()
    df_filtered = df[df["biomass_id"].isin(valid_ids)]
    pivot_df = df_filtered.pivot_table(index=["biomass_id", "biomass_name"], columns="property", values="ar_value", aggfunc="first").reset_index()
    for prop in required_properties:
        pivot_df = pivot_df[pivot_df[prop].notna()]
    final_columns = ["biomass_id", "biomass_name"] + required_properties
    result_df = pivot_df[final_columns]
    result_df = result_df.sort_values(by="biomass_id")
    output_path = 'C:/Users/asus/Desktop/spark/sparkenv/final_filtered_csv.csv'
    result_df.to_csv(output_path, index=False)
    print("Final DataFrame has been saved to:", output_path)
    return result_df

def drop_low_correlation_columns(dataframe, target_column, threshold=0.2):
    correlation_matrix = dataframe.corr()
    target_correlation = correlation_matrix[target_column]
    columns_to_drop = target_correlation[abs(target_correlation) < threshold].index.tolist()
    filtered_dataframe = dataframe.drop(columns=columns_to_drop)
    return filtered_dataframe

def generate_random_name():
    letters = string.ascii_lowercase
    return ''.join(random.choices(letters, k=6)) + ' ' + ''.join(random.choices(letters, k=6))

required_properties = ["Moisture content", "Ash content", "Volatile matter", "Fixed carbon", "Carbon", "Hydrogen", "Oxygen", "Nitrogen", "Net calorific value (LHV)"]
subset_df = df[required_properties]

correlation_matrix = subset_df.corr()



target_column = "Net calorific value (LHV)"
filtered_df = drop_low_correlation_columns(subset_df, target_column, threshold=0.2)

print("Columns after dropping low correlation ones:")
print(filtered_df.columns)

features = filtered_df.columns.drop(target_column).tolist()
X = filtered_df[features]
y = filtered_df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

new_data = pd.DataFrame(columns=features)
num_new_rows = 2000000
new_rows_list = []

for i in tqdm(range(num_new_rows), desc="Generating rows"): #generating rows with the same mean and std
    new_row = {}
    new_row['ID'] = i + 1
    new_row['name'] = generate_random_name()
    for feature in features:
        mean_feature = df[feature].mean()
        mean_std = df[feature].std()
        new_row[feature] = abs(mean_feature + np.random.normal(0, mean_std))
    new_rows_list.append(new_row)

new_data = pd.DataFrame(new_rows_list)

for feature in features: #compare mean and std of old and new rows
    original_mean = df[feature].mean()
    original_std = df[feature].std()
    new_mean = new_data[feature].mean()
    new_std = new_data[feature].std()
    
    print(f"Original {feature} - mean: {original_mean}, std: {original_std}")
    print(f"New {feature} - mean: {new_mean}, std: {new_std}")

new_predictions = model.predict(new_data[features])
new_data["Net calorific value (LHV)"] = new_predictions

mse_new = mean_squared_error(new_data[target_column], new_predictions)
r2_new = r2_score(new_data[target_column], new_predictions)

print(f"Mean Squared Error on new data: {mse_new}")
print(f"R^2 Score on new data: {r2_new}")

output_path = 'C:/Users/asus/Desktop/spark/sparkenv/generated_800k_data.csv'
new_data.to_csv(output_path, index=False)
print("Generated 5m DataFrame has been saved to:", output_path)

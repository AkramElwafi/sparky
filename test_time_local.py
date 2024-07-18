import os
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import tqdm
from pyspark.sql.window import Window

os.environ['PYSPARK_PYTHON'] = 'python'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python'


spark = SparkSession.builder \
    .appName('ComplexComputation') \
    .master('local[3]') \
    .getOrCreate()

start_time = time.time()


custom_schema = StructType([
    StructField("biomass_id", FloatType(), True),
    StructField("biomass_name", StringType(), True),
    StructField("Moisture content", FloatType(), True),
    StructField("Volatile matter", FloatType(), True),
    StructField("Fixed carbon", FloatType(), True),
    StructField("Carbon", FloatType(), True),
    StructField("Hydrogen", FloatType(), True),
    StructField("Net calorific value (LHV)", FloatType(), True),
])

output_path = r'sparky/build/workspace/data/generated_2millions_data.csv'
df = spark.read.csv(output_path, header=True, schema=custom_schema)

df = df.drop('biomass_id', 'biomass_name')

columns = df.columns

stats = df.agg(
    *[F.mean(col).alias(f'Mean_{col}') for col in columns],
    *[F.stddev(col).alias(f'StdDev_{col}') for col in columns],
    *[F.min(col).alias(f'Min_{col}') for col in columns],
    *[F.max(col).alias(f'Max_{col}') for col in columns],
    *[F.expr(f'percentile_approx(`{col}`, 0.25)').alias(f'25th_Percentile_{col}') for col in columns],
    *[F.expr(f'percentile_approx(`{col}`, 0.75)').alias(f'75th_Percentile_{col}') for col in columns],
    *[F.kurtosis(col).alias(f'Kurtosis_{col}') for col in columns],
    *[F.skewness(col).alias(f'Skewness_{col}') for col in columns]
)

stats.show()

windowSpec = Window.rowsBetween(-10000, 0)
expandingWindowSpec = Window.orderBy("Moisture content").rowsBetween(Window.unboundedPreceding, Window.currentRow)
orderedWindowSpec = Window.orderBy("Moisture content")

for col_name in tqdm.tqdm(columns):
    for _ in range(10):
        df = df.withColumn(f'{col_name}_rolling_mean', F.avg(col(col_name)).over(windowSpec))
        df = df.withColumn(f'{col_name}_rolling_std', F.stddev(col(col_name)).over(windowSpec))
        df = df.withColumn(f'{col_name}_expanding_mean', F.avg(col(col_name)).over(expandingWindowSpec))
        df = df.withColumn(f'{col_name}_expanding_std', F.stddev(col(col_name)).over(expandingWindowSpec))

for col_name in tqdm.tqdm(columns):
    for _ in range(10):
        df = df.withColumn(f'{col_name}_log', F.log1p(col(col_name)))
        df = df.withColumn(f'{col_name}_sqrt', F.sqrt(col(col_name)))
        df = df.withColumn(f'{col_name}_cumsum', F.sum(col(col_name)).over(expandingWindowSpec))
        df = df.withColumn(f'{col_name}_cumprod', F.exp(F.sum(F.log(col(col_name) + 1)).over(expandingWindowSpec)))
        df = df.withColumn(f'{col_name}_exp', F.exp(col(col_name)))
        df = df.withColumn(f'{col_name}_sin', F.sin(col(col_name)))
        df = df.withColumn(f'{col_name}_cos', F.cos(col(col_name)))
        df = df.withColumn(f'{col_name}_tan', F.tan(col(col_name)))
        df = df.withColumn(f'{col_name}_diff', col(col_name) - F.lag(col(col_name), 1).over(orderedWindowSpec))
        df = df.withColumn(f'{col_name}_pct_change', (col(col_name) - F.lag(col(col_name), 1).over(orderedWindowSpec)) / F.lag(col(col_name), 1).over(orderedWindowSpec))

grouped_stats = df.groupBy((F.floor(F.monotonically_increasing_id() / 100)).alias("group")).agg(
    F.mean('Moisture content').alias('Mean_Moisture content'),
    F.stddev('Moisture content').alias('StdDev_Moisture content'),
    F.min('Moisture content').alias('Min_Moisture content'),
    F.max('Moisture content').alias('Max_Moisture content'),
    F.mean('Volatile matter').alias('Mean_Volatile matter'),
    F.stddev('Volatile matter').alias('StdDev_Volatile matter'),
    F.min('Volatile matter').alias('Min_Volatile matter'),
    F.max('Volatile matter').alias('Max_Volatile matter'),
    F.mean('Fixed carbon').alias('Mean_Fixed carbon'),
    F.stddev('Fixed carbon').alias('StdDev_Fixed carbon'),
    F.min('Fixed carbon').alias('Min_Fixed carbon'),
    F.max('Fixed carbon').alias('Max_Fixed carbon'),
    F.mean('Carbon').alias('Mean_Carbon'),
    F.stddev('Carbon').alias('StdDev_Carbon'),
    F.min('Carbon').alias('Min_Carbon'),
    F.max('Carbon').alias('Max_Carbon'),
    F.mean('Hydrogen').alias('Mean_Hydrogen'),
    F.stddev('Hydrogen').alias('StdDev_Hydrogen'),
    F.min('Hydrogen').alias('Min_Hydrogen'),
    F.max('Hydrogen').alias('Max_Hydrogen'),
    F.mean('Net calorific value (LHV)').alias('Mean_Net calorific value (LHV)'),
    F.stddev('Net calorific value (LHV)').alias('StdDev_Net calorific value (LHV)'),
    F.min('Net calorific value (LHV)').alias('Min_Net calorific value (LHV)'),
    F.max('Net calorific value (LHV)').alias('Max_Net calorific value (LHV)')
)

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
from pyspark.sql import SparkSession
from pyspark.sql.functions import corr
from itertools import combinations
from pyspark.sql.functions import col
import time
import os
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql.window import Window

# Initialize Spark session
spark = SparkSession.\
    builder.\
    appName("pyspark-notebook").\
    master("spark://spark-master-2c:7077").\
    config("spark.executor.memory", "1g").\
    config("spark.executor.cores", "1").\
    config("spark.driver.memory", "1g").\
    getOrCreate()

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

input_path = 'hdfs://namenode:8020/data/generated_2mill_data.csv'
df = spark.read.csv(input_path, header=True, schema=custom_schema)

df = df.drop('biomass_id', 'biomass_name')

columns = df.columns

# Calculate statistics
stats = df.agg(
    *[F.mean(col).alias('Mean_{}'.format(col)) for col in columns],
    *[F.stddev(col).alias('StdDev_{}'.format(col)) for col in columns],
    *[F.min(col).alias('Min_{}'.format(col)) for col in columns],
    *[F.max(col).alias('Max_{}'.format(col)) for col in columns],
    *[F.expr('percentile_approx(`{}`, 0.25)'.format(col)).alias('25th_Percentile_{}'.format(col)) for col in columns],
    *[F.expr('percentile_approx(`{}`, 0.75)'.format(col)).alias('75th_Percentile_{}'.format(col)) for col in columns],
    *[F.kurtosis(col).alias('Kurtosis_{}'.format(col)) for col in columns],
    *[F.skewness(col).alias('Skewness_{}'.format(col)) for col in columns]
)

stats.show()
windowSpec = Window.rowsBetween(-10000, 0)
expandingWindowSpec = Window.orderBy("Moisture content").rowsBetween(Window.unboundedPreceding, Window.currentRow)
orderedWindowSpec = Window.orderBy("Moisture content")
print("ttt1")
for col_name in columns:
    for _ in range(10):
        df = df.withColumn('{} rolling_mean'.format(col_name), F.avg(col(col_name)).over(windowSpec))
        df = df.withColumn('{} rolling_std'.format(col_name), F.stddev(col(col_name)).over(windowSpec))
        df = df.withColumn('{} expanding_mean'.format(col_name), F.avg(col(col_name)).over(expandingWindowSpec))
        df = df.withColumn('{} expanding_std'.format(col_name), F.stddev(col(col_name)).over(expandingWindowSpec))
print("ttt2")

for col_name in columns:
    for _ in range(10):
        df = df.withColumn('{} log'.format(col_name), F.log1p(col(col_name)))
        df = df.withColumn('{} sqrt'.format(col_name), F.sqrt(col(col_name)))
        df = df.withColumn('{} cumsum'.format(col_name), F.sum(col(col_name)).over(expandingWindowSpec))
        df = df.withColumn('{} cumprod'.format(col_name), F.exp(F.sum(F.log(col(col_name) + 1)).over(expandingWindowSpec)))
        df = df.withColumn('{} exp'.format(col_name), F.exp(col(col_name)))
        df = df.withColumn('{} sin'.format(col_name), F.sin(col(col_name)))
        df = df.withColumn('{} cos'.format(col_name), F.cos(col(col_name)))
        df = df.withColumn('{} tan'.format(col_name), F.tan(col(col_name)))
        df = df.withColumn('{} diff'.format(col_name), col(col_name) - F.lag(col(col_name), 1).over(orderedWindowSpec))
        df = df.withColumn('{} pct_change'.format(col_name), (col(col_name) - F.lag(col(col_name), 1).over(orderedWindowSpec)) / F.lag(col(col_name), 1).over(orderedWindowSpec))
print("ttt3")

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
print("ttt4")

end_time = time.time()
exec_time=end_time-start_time
print("Total execution time: ",exec_time ,"seconds")

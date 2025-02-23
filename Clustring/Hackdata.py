from pyspark.sql import SparkSession

input_path_hackdata = '../rawdata/kmeans/hack_data.csv'

spark_kmeans_hackdata = SparkSession.builder.appName('KMeans_hackdata').getOrCreate()

df_kmeans_hack_data = spark_kmeans_hackdata.read.format('csv') \
    .option('inferSchema', True) \
    .option('header', True) \
    .load(input_path_hackdata)

# df_kmeans_hack_data.show(3)

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

df_kmeans_hack_data.columns

# Print the columns
print("Print the columns: ", df_kmeans_hack_data.columns)

# drop Location (string) column
hack_data_cols = ['Session_Connection_Time', 'Bytes Transferred', 'Kali_Trace_Used',
                  'Servers_Corrupted', 'Pages_Corrupted', 'WPM_Typing_Speed']

assembler_hackdata = VectorAssembler(inputCols=hack_data_cols, outputCol='features')

final_df_kmeans_hack_data = assembler_hackdata.transform(df_kmeans_hack_data)

# features columns is added, confirm the same
final_df_kmeans_hack_data.printSchema()

# Scaling the dataset

from pyspark.ml.feature import StandardScaler

scaler_hack_data = StandardScaler(inputCol='features', outputCol='scaledFeatures')

scaler_hack_data_model = scaler_hack_data.fit(final_df_kmeans_hack_data)

cluster_final_hack_data = scaler_hack_data_model.transform(final_df_kmeans_hack_data)

# scaledFeatures columns is added, confirm the same
cluster_final_hack_data.printSchema()

kmeans2 = KMeans(featuresCol='scaledFeatures', k=2)
kmeans3 = KMeans(featuresCol='scaledFeatures', k=3)

# Trains a k-means model.
model_kmeans_hackdata_2 = kmeans2.fit(cluster_final_hack_data)
model_kmeans_hackdata_3 = kmeans3.fit(cluster_final_hack_data)

model_kmeans_hackdata_2.transform(cluster_final_hack_data).show()

model_kmeans_hackdata_2.transform(cluster_final_hack_data).groupBy('prediction').count().show()

model_kmeans_hackdata_3.transform(cluster_final_hack_data).groupBy('prediction').count().show()


spark_kmeans_hackdata.stop()

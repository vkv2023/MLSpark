from pyspark.sql import SparkSession

"""
Seven geometric parameters of wheat kernels were measured:
area A, perimeter P, compactness C = 4piA/P^2, length of kernel, 
width of kernel, asymmetry coefficient, length of kernel groove. 
All of these parameters were real-valued continuous.
"""
spark_seed = SparkSession.builder.appName('seeds').getOrCreate()

inout_path_seed = '../rawdata/kmeans/seeds_dataset.csv'

df_seed = spark_seed.read.format('csv')\
    .option("inferSchema",True)\
    .option("header",True)\
    .load(inout_path_seed)

# df_seed.show(2)

# for row in df_seed.head(5):
#     print(row,"\n")

"""
Format the Data
"""

from pyspark.ml.feature import VectorAssembler

assembler_seeds = VectorAssembler(inputCols=df_seed.columns, outputCol='features')
# Final data set will be passed in model
final_df_seed = assembler_seeds.transform(df_seed)

from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans

scaler_seed_data = StandardScaler(inputCol='features', outputCol='scaledFeatures')
# scaler_seed_data = StandardScaler(inputCol='features', outputCol='scaledFeatures', withStd=True, withMean=False)

# Compute summary statistics by fitting the StandardScaler
model_seed = scaler_seed_data.fit(final_df_seed)

# Normalize each feature to have unit standard deviation
final_df_seed_data = model_seed.transform(final_df_seed)

"""
Training the Model and Evaluate
"""
# Training a k-means model
kmeans_seed_3 = KMeans(featuresCol='scaledFeatures', k=3)
kmeans_seed_3_model = kmeans_seed_3.fit(final_df_seed_data)

# Evaluate clustering by computing Within Set Sum of Squared Errors.
# wssse_seed = kmeans_seed_3_model.computeCost(final_df_seed)
# print("Within Set Sum of Squared Errors = " + str(wssse_seed))

kmeans_seed_3_model.transform(final_df_seed_data).show(2)

# group the data based upon clusters
kmeans_seed_3_model.transform(final_df_seed_data).groupBy('prediction').count().show()


print("Stopping the SparkSession...")

# Stop the sparksession
spark_seed.stop()




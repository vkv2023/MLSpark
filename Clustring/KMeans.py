from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans


inout_path_KMeans = '../rawdata/sample_kmeans_data.txt'

spark_kmeans = SparkSession.builder.appName('KMeans').getOrCreate()

df_kmeans = spark_kmeans.read.format('libsvm')\
    .option("inferSchema",True)  \
    .option("header", True)  \
    .load(inout_path_KMeans)

df_kmeans.show()

print("*"*50)

final_df_kmenas = df_kmeans.select('features')
final_df_kmenas.show()

print("*"*50)

# Lets set a K value as 2
# kmeans = KMeans().setK(2).setSeed(1)
# set seed value of center, lets put it 1 for now
# Lets set a K value as 2
kmeans = KMeans().setK(3).setSeed(1)

# Trains a k-means model.
model_kmeans = kmeans.fit(final_df_kmenas)

# within set sum of squared error
# wssse = model_kmeans.computeCost(final_df_kmenas)
# print(wssse)

print("*"*50)
# Make predictions
predictions_kmeans = model_kmeans.transform(final_df_kmenas)
predictions_kmeans.show()

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

print("*"*50)

silhouette = evaluator.evaluate(predictions_kmeans)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# Shows the result.
centers = model_kmeans.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

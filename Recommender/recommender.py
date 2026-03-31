from pyspark.sql import SparkSession

from pyspark1.LinearRegression.LinearRegression import training

spark_rec = SparkSession.builder.appName('rec').getOrCreate()

input_path_hackdata = '../rawdata/movielens/ratings.csv'

from pyspark.ml.recommendation import ALS

from pyspark.ml.evaluation import RegressionEvaluator

data = spark_rec.read.format("csv").option('inferSchema',True)\
        .option('header',True)\
        .load(input_path_hackdata)

data.show()

data.describe().show()

training_rec, test_rec = data.randomSplit([0.7, 0.3], seed=42)

# Create ALS

als = ALS(maxIter=5, regParam=0.01, userCol='userId', itemCol='movieId', ratingCol='rating', coldStartStrategy="drop")

model_als = als.fit(training_rec)

prediction_rec = model_als.transform(test_rec)

prediction_rec.show(5)

# rec_evaluator = RegressionEvaluator(metricName='rmse', labelCol= 'rating', predictionCol='prediction')
# rmse = rec_evaluator.evaluate(prediction_rec)
# print('RMSE  :', rmse)


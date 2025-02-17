from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
import numpy as np

spark = SparkSession.builder\
    .appName('LinerRegression')\
    .getOrCreate()

input_path = 'rawdata\\sample_linear_regression_data.txt'

training = spark.read.format('libsvm')\
    .option("header", True)\
    .option("inferSchema",True)\
    .load(input_path)

training.printSchema()
training.show(4)

#Lineer Regression
# Creating an instance of the model
lr = LinearRegression(featuresCol='features', labelCol='label', predictionCol='prediction')
# lr1 = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# fit and train the model
lrModel = lr.fit(training)

# Set up the summary attribute of the machine learning Model
training_summary = lrModel.summary

# We can call many methods on summary
training_summary.rootMeanSquaredError


# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))
print("Training Summary: %s" % str(training_summary.rootMeanSquaredError))

# Creating a one more data from
all_data = spark.read.format('libsvm')\
    .option("header",True)\
    .option("inferSchema",True)\
    .load(input_path)

#  Lets split the data in some training and test data randomly
# split_object = all_data.randomSplit([0.7,0.3])
train_data, test_data = all_data.randomSplit([0.7,0.3])

# lets see the data
print('\ntrain data .........\n')
train_data.describe().show()
print('test data .........\n')
test_data.describe().show()


# lets create one more model and check if that model \
# fits correct on out test model data

correct_model = lr.fit(train_data)
test_summary = correct_model.evaluate(test_data)
test_summary.rootMeanSquaredError

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(correct_model.coefficients))
print("Intercept: %s" % str(correct_model.intercept))
print("Training Summary: %s" % str(test_summary.rootMeanSquaredError))


unlabeled_data = test_data.select('features')
unlabeled_data.show()

predictions = correct_model.transform(unlabeled_data)
predictions.show()
from shutil import which
from typing import final

from pyspark.sql import SparkSession

from LinearRegression import unlabeled_data, predictions

spark = SparkSession.builder \
    .appName("LinearRegression_Ecom_cust") \
    .getOrCreate()

input_path = '../rawdata/Ecommerce_Customers.csv'

ecom_df = spark.read.format('csv') \
    .option('inferSchema', True) \
    .option('header', True) \
    .load(input_path)

from pyspark.ml.regression import LinearRegression

# Show Print schema
ecom_df.printSchema()

# for row in ecom_df.head(5):
#     print(row)

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

"""
The assembler object uses this input columns to convert to a features column
which we can use to train our model.
"""
assembler = VectorAssembler(inputCols=['Avg Session Length', 'Time on App',
                                       'Time on Website', 'Length of Membership'],
                            outputCol='features')

output = assembler.transform(ecom_df)

# output.head()

final_data = output.select('features', 'Yearly Amount Spent')

# final_data.show()

train_data, test_data = final_data.randomSplit([0.7, 0.3])

train_data.describe().show()

test_data.describe().show()

lr = LinearRegression(labelCol='Yearly Amount Spent', featuresCol='features')

lr_model = lr.fit(train_data)

test_summary = lr_model.evaluate(test_data)

# residuals are the difference between predicted values and actual labeled from test data
test_summary.residuals.show()

print("\nrootMeanSquaredError: ", str(test_summary.rootMeanSquaredError))

print("\ntest_summary: ", str(test_summary.r2))

final_data.describe().show()

# How to use the prediction or deployed model for any unlabled data

unlabeled_data = test_data.select('features')

unlabeled_data.show()

predictions = lr_model.transform(unlabeled_data)

predictions.show()

print('Stopping spark session.....')
# Stop the SparkSession
spark.stop()

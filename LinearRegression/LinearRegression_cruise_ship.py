from pyspark.sql import SparkSession

from pyspark1.LinearRegression.LinearRegression_ecom_customer import final_data
from pyspark1.LogisticRegression.LogisticRegression_titanic import train_data

spark = SparkSession.builder.appName("cruise").getOrCreate()

input_path = '../rawdata/cruise_ship_info/cruise_ship_info.csv'

df = spark.read.format("csv")\
    .option("header",True)\
    .option("inferSchema",True)\
    .load(input_path)


df.printSchema()

df.groupBy('Cruise_line').count().show()

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

# converting Cruise_line to category numerical values
Cruise_indexer = StringIndexer(inputCol='Cruise_line', outputCol='Cruise_category')
# Cruise_encoder = OneHotEncoder(inputCol='Cruise_category', outputCol='CruiseVector')

indexed = Cruise_indexer.fit(df).transform(df)
indexed.show(3)

# indexed.columns
assembler = VectorAssembler(inputCols=['Age','Tonnage','passengers','length','cabins','passenger_density']
                            , outputCol='features')
output = assembler.transform(indexed)
output.select('features', 'crew').show()

final_data_crew = output.select(['features', 'crew'])
train_data_crew, test_data_crew = final_data_crew.randomSplit([0.7, 0.3])

#  Lets use regression
from pyspark.ml.regression import LinearRegression

# train_data_crew.describe()
# test_data_crew.describe()

ship_lr = LinearRegression(labelCol='crew')

trained_ship_model = ship_lr.fit(train_data_crew)

ship_results = trained_ship_model.evaluate(test_data_crew)

# See the prediction on the test_Data
ship_results.predictions.show(20)

# See the RootMean square value
print("\nrootMeanSquaredError: ", str(ship_results.rootMeanSquaredError))

# See the R2 square value
print("\nR2: ", str(ship_results.r2))

# See the Mean absolute error value
print("\nrootMeanSquaredError: ", str(ship_results.meanSquaredError))


from pyspark.sql.functions import corr

# select correlation with different columns along with crew

df.describe('crew','passenger_density').show()

df.select(corr('crew','passenger_density')).show()


print('Stopping spark session.....')
# Stop the SparkSession
spark.stop()



from pyspark.sql import SparkSession


# Initialize Spark session and context
spark = (SparkSession \
         .builder \
         .appName("SimplePySparkJob") \
         .config("spark.eventlog.enabled", "true") \
         .config("spark.logConf", "true") \
         .getOrCreate())

# get the logger



# Read a CSV file into a DataFrame
input_file = "rawdata\\online-retail-dataset.csv"

retail_df=spark.read.format('csv')\
    .option("inferSchema",True)\
    .option("header",True)\
    .load(input_file)
# Shows the schema of the dataframe
retail_df.printSchema()
# shows the dataframe
retail_df.show()
#explain shows the lineage
retail_df.sort('UnitPrice').explain()
# shows the data after sorting on UnitPrice
retail_df.sort('UnitPrice').show(5)

for row in retail_df.head(5):
    print(row, '\n')

retail_df.columns
retail_df.select('InvoiceNo','StockCode').describe().show()

# Stopping spark session
print('Stopping spark session.....')
spark.stop()

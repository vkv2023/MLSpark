from pyspark.sql import SparkSession

input_path_decisionTreeData = '../rawdata/College.csv'

spark_decisionTreeData = SparkSession.builder.appName("DecisionTree").getOrCreate()

decisionTreeData = spark_decisionTreeData.read.format("csv")\
    .option('inferSchema', True)\
    .option('header', True)\
    .load(input_path_decisionTreeData)


# decisionTreeData.printSchema()

# decisionTreeData.show(2)

from pyspark.ml.feature import VectorAssembler

print(decisionTreeData.columns)

assembler_decisionTree = VectorAssembler(inputCols=['Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F_Undergrad', 'P_Undergrad', 'Outstate', 'Room_Board', 'Books', 'Personal', 'PhD', 'Terminal', 'S_F_Ratio', 'Perc_Alumni', 'Expend', 'Grad_Rate'],
                                         outputCol='features')

final_df_decisiontree = assembler_decisionTree.transform(decisionTreeData)

from pyspark.ml.feature import StringIndexer

indexer_ = StringIndexer(inputCol='Private', outputCol='PrivateIndex')

output_fixed = indexer_.fit(final_df_decisiontree).transform(final_df_decisiontree)

output_fixed.printSchema()

final_df_decisiontree_data = output_fixed.select('features', 'PrivateIndex')

train_data_d, test_data_d = final_df_decisiontree_data.randomSplit([0.7, 0.3])

from pyspark.ml.classification import (DecisionTreeClassifier, GBTClassifier, RandomForestClassifier)



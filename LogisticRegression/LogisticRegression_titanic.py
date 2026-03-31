from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder\
    .appName("LogisticRegression_titanic")\
    .getOrCreate()

input_path_train = '../rawdata/titanic/train.csv'
input_path_test = '../rawdata/titanic/test.csv'
input_path_submitted = '../rawdata/titanic/gender_submission.csv'

"""
Collect the data in train dataframe
"""

train_df = spark.read.format('csv')\
    .option('inferSchema',True)\
    .option('header',True)\
    .load(input_path_train)
"""
Collect the data in test dataframe
"""

test_df = spark.read.format('csv')\
    .option('inferSchema',True)\
    .option('header',True)\
    .load(input_path_test)

"""
Collect the data in the dataframe for which model to be applied
"""

submitted_df = spark.read.format('csv')\
    .option('inferSchema',True)\
    .option('header',True)\
    .load(input_path_submitted)

"""
selecting the columns required for our analysis and dropping the nulls
"""

train_data = train_df.select(['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
test_data = test_df.select(['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])

"""
drop null from the final data frames
"""
train_data_final = train_data.dropna()
test_data_final = test_data.dropna()

"""
Sex and Embarked columns are encoded from the index we are creating from Stringindexer. 
The assembler object uses this embarked vector to convert to a features column 
which we can use to train our model.
"""

gender_index = StringIndexer(inputCol='Sex', outputCol='SexIndex')
gender_encoder = OneHotEncoder(inputCol='SexIndex',outputCol='SexVector')

embark_index = StringIndexer(inputCol='Embarked', outputCol='EmbarkedIndex')
embark_encoder = OneHotEncoder(inputCol='EmbarkedIndex',outputCol='EmbarkedVector')

"""
Assembling a feature column Vector on different columns  
"""

assembler = VectorAssembler(inputCols=['Pclass','SexVector','Age','SibSp','Parch','Fare','EmbarkedVector']
                            , outputCol='features')

"""
Creating logistic regression and labeling as Survived
"""
lg = LogisticRegression(featuresCol='features',labelCol='Survived')

"""
different steps were created to provide to the pipeline as stages. 
Which is ready to accept the training data directly
"""

pl = Pipeline(stages=[gender_index,gender_encoder,embark_index,embark_encoder,assembler])

"""
using pl object to fit and transform our data in a single line. 
The trained model is created after fitting the training data. 
up next, we will use this model to test and predict survivors.
"""
assembled_train_data = pl.fit(train_data_final)\
                        .transform(train_data_final)

assembled_test_data = pl.fit(test_data_final)\
                       .transform(test_data_final)

"""
we are using our pipeline to give us transformed data which will have features required 
for testing. This will also index, encode and vector assemble our data as we saw previously
"""

lg_model = lg.fit(assembled_train_data)

results = lg_model.transform(assembled_test_data)

results_with_survived = results.join(submitted_df,['PassengerId'])\
    .select('PassengerId','Survived','prediction')

"""
The test data does not come with the actual value of survivors. 
So, we have added PassengerId before when selecting so that we can join 
with the gender_submission CSV file. 
We are reading the CSV file and joining on PassengerId to get the survivors 
column which we can use for our binary classifier next.
"""

evaluation = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='Survived')

AUC = evaluation.evaluate(results_with_survived)

results_with_survived.select('Survived','prediction').show()

print("\nPrinting AUC value:", AUC, '\n')

print('Stopping spark session.....')
# Stop the SparkSession
spark.stop()







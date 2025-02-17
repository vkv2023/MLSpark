from pyexpat import features
from typing import List
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation
from pysparkfilereader import spark

data = Vectors[
               (Vectors.sparse(4,[(0,1.0), (3,-2.0)]),),
               (Vectors.dense([4.0,5.0,.0,3.0]),),
               (Vectors.dense([6.0,7.0,.0,8.0]),),
               (Vectors.sparse(4, [(0,9.0), (3, 1.0)]),)
               ]
df = spark.createDataFrame(data, ["features"])

r1 = Correlation.corr(df,"features").head()

print("Pearson Correlation Matrix :\n",str(r1[0]))

r2 = Correlation.corr(df,"features","spearman").head()

print("Spearman Correlation Matrix :\n",str(r2[0]))

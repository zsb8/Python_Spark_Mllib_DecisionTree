# Python_Spark_DecisionTree
use DecisionTree and BinaryClassificationMetrics to find evergreen. 

Running environment is Spark + Hadoop + PySpark    
Used the algorithm is DecisionTree.     
Used the library is pyspark.mllib.    

# Stage1:  Read data
Placed the tsv on hadoop. Built 3 data sets: (1) Train data, (2) Validation data, (3) Test data.


# Draw the picture
## Compare the impurities    
Impurity is the paramter of DecisionTree. I used 'gini' and 'entropy'. Compared the results from them. Found use 'entropy' was better.
![image](https://user-images.githubusercontent.com/75282285/192569344-5a66ba9f-4438-4e62-99c8-103a0e5433a7.png)


# Stage2: Train and evaluate   
Created the model using train data set.   
Calculated the AUC using validation data set.
Sorted the metrics.    
Found the best parameters includ the best AUC and the best model.   

# Stage3: Test
Used the test data set and the best model to calculate the AUC. If testing AUC is similare as the best AUC, it is OK.







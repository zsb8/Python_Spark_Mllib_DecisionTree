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
~~~
    impurity_list = ['gini', 'entropy']
    max_depth_list = [10]
    max_bins_list = [10]
    my_metrics = [
        train_evaluation_model(train_d, validation_d, impurity, max_depth, max_bins)
        for impurity in impurity_list
        for max_depth in max_depth_list
        for max_bins in max_bins_list
    ]
    df = pd.DataFrame(my_metrics,
                      index=impurity_list,
                      columns=['AUC', 'duration', 'impurity', 'maxDepth', 'maxBins', 'model'])
    show_chart(df, 'impurity', 'AUC', 'duration')
~~~
![image](https://user-images.githubusercontent.com/75282285/192569344-5a66ba9f-4438-4e62-99c8-103a0e5433a7.png)

Compared the different depth parameters. It looked use 10 as maxPath, the AUC was the best one, the cost time was not hight than others. 
~~~
    impurity_list = ['entropy']
    max_depth_list = [3, 5, 10, 15, 20, 25]
    max_bins_list = [10]
    my_metrics = [
        train_evaluation_model(train_d, validation_d, impurity, max_depth, max_bins)
        for impurity in impurity_list
        for max_depth in max_depth_list
        for max_bins in max_bins_list
    ]
    df = pd.DataFrame(my_metrics,
                      index=max_depth_list,
                      columns=['AUC', 'duration', 'impurity', 'maxDepth', 'maxBins', 'model'])
    show_chart(df, 'maxDepth', 'AUC', 'duration')
~~~
![image](https://user-images.githubusercontent.com/75282285/192575887-816a90e3-d786-4300-9932-e17c247371e2.png)






# Stage2: Train and evaluate   
Created the model using train data set.   
Calculated the AUC using validation data set.
Sorted the metrics.    
Found the best parameters includ the best AUC and the best model.   

# Stage3: Test
Used the test data set and the best model to calculate the AUC. If testing AUC is similare as the best AUC, it is OK.







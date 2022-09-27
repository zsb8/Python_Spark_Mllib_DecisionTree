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

Compared the difference of depth parameters. It looked like the AUC was the best one if I use maxDepth=10 this time, the cost time was not highter than others. 
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

Compared the difference of maxBins parameters. It looked like the AUC was the best one if use maxBins=200 this time.
~~~
    impurity_list = ['entropy']
    max_depth_list = [10]
    max_bins_list = [3, 5, 10, 50, 100, 200]
    my_metrics = [
        train_evaluation_model(train_d, validation_d, impurity, max_depth, max_bins)
        for impurity in impurity_list
        for max_depth in max_depth_list
        for max_bins in max_bins_list
    ]
    df = pd.DataFrame(my_metrics,
                      index=max_bins_list,
                      columns=['AUC', 'duration', 'impurity', 'maxDepth', 'maxBins', 'model'])
    show_chart(df, 'maxBins', 'AUC', 'duration')
~~~
![image](https://user-images.githubusercontent.com/75282285/192578482-30a08976-e265-4500-9e18-c1f5e3041344.png)



# Stage2: Train and evaluate   
Created the model using train data set.   
Calculated the AUC using validation data set.
Sorted the metrics.    
Found the best parameters includ the best AUC and the best model.   
![image](https://user-images.githubusercontent.com/75282285/192605629-baa98d8f-39d9-423d-be10-32bca9cb4861.png)



# Stage3: Test
Used the test data set and the best model to calculate the AUC. If testing AUC is similare as the best AUC, it is OK.
![image](https://user-images.githubusercontent.com/75282285/192605683-12852eb4-7343-4afd-8a88-3a61e6083259.png)



![image](https://user-images.githubusercontent.com/75282285/192587362-ac4c79f9-f87c-4da9-9acc-b67412eb2fa5.png)


![image](https://user-images.githubusercontent.com/75282285/192587799-e3b653f6-4d73-4b33-8126-a1debb838366.png)



![image](https://user-images.githubusercontent.com/75282285/192587445-b66c945a-929d-4b42-80c5-5ab5df2d35c1.png)








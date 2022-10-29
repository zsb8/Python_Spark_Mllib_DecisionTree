Data is pubic.
# Python_Spark_DecisionTree
Use DecisionTree and BinaryClassificationMetrics to find evergreen webside. 

Running environment is Spark + Hadoop + PySpark    
Used the algorithm is DecisionTree.     
Used the library is pyspark.mllib.    

# Stage1:  Read data
Placed the tsv on hadoop. Built 3 data sets: (1) Train data, (2) Validation data, (3) Sub_test data.


## Compare the parameters
Impurity is the paramter of DecisionTree. I used 'gini' and 'entropy'. Compared the results from them. Found use 'entropy' was better.
~~~python
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
~~~python
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
~~~python
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
Used the sub_test data set and the best model to calculate the AUC. If testing AUC is similare as the best AUC, it is OK.
![image](https://user-images.githubusercontent.com/75282285/192605683-12852eb4-7343-4afd-8a88-3a61e6083259.png)


# Stage4: Predict
Use the test data (in Hadoop, test.tsv) and the model (calculated after Stage2) to predict.
~~~python
def predict_data(best_model):
    raw_data_with_header = sc.textFile(path + "test.tsv")
    header = raw_data_with_header.first()
    raw_data = raw_data_with_header.filter(lambda x: x != header)
    r_data = raw_data.map(lambda x: x.replace("\"", ""))
    lines_test = r_data.map(lambda x: x.split('\t'))
    data_rdd = lines_test.map(lambda x: (x[0], extract_features(x, categories_map, len(x))))
    dic_desc = {
        0: 'temp web',
        1: 'evergreen web'
    }
    for data in data_rdd.take(10):
        result_predict = best_model.predict(data[1])
        print(f"web:{data[0]}, \n predict:{result_predict}, desc: {dic_desc[result_predict]}")
~~~
![image](https://user-images.githubusercontent.com/75282285/192613552-9d77a401-1667-47ac-9e7e-1f8725e15dbc.png)


# Spark monitor
![image](https://user-images.githubusercontent.com/75282285/192587362-ac4c79f9-f87c-4da9-9acc-b67412eb2fa5.png)
![image](https://user-images.githubusercontent.com/75282285/192587799-e3b653f6-4d73-4b33-8126-a1debb838366.png)
![image](https://user-images.githubusercontent.com/75282285/192587445-b66c945a-929d-4b42-80c5-5ab5df2d35c1.png)

# DebugString
Print the DebugString.
~~~python
print(model.toDebugString())
~~~

![image](https://user-images.githubusercontent.com/75282285/192617611-b294921c-5be5-4393-9073-96793e3c46b4.png)






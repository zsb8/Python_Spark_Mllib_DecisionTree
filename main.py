from pyspark import SparkConf, SparkContext
import numpy as np
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from time import time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def create_spark_context():
    global sc, path
    sc = SparkContext(conf=SparkConf().setAppName('test'))
    path = "hdfs://node1:8020/input/"


def read_data():
    global lines, categories_map
    raw_data_with_header = sc.textFile(path + "train.tsv")
    # print(f"raw_data_with_Header=={raw_data_with_header.take(2)}")
    header = raw_data_with_header.first()
    raw_data = raw_data_with_header.filter(lambda x: x != header)
    r_data = raw_data.map(lambda x: x.replace("\"", ""))
    lines = r_data.map(lambda x: x.split('\t'))
    categories_map = lines.map(lambda fields: fields[3]).distinct().zipWithIndex().collectAsMap()


def convert_float(x):
    if x == "?":
        result = 0
    else:
        result = float(x)
    return result


def extract_features(field, categories_map, feature_end):
    category_idx = categories_map[field[3]]
    category_features = np.zeros(len(categories_map))
    category_features[category_idx] = 1
    numerical_features = [convert_float(field) for field in field[4: feature_end]]
    result = np.concatenate((category_features, numerical_features))
    return result


def extract_label(field):
    label = field[-1]
    result = float(label)
    return result


def prepare_data():
    labelpoint_RDD = lines.map(lambda r: LabeledPoint(extract_label(r), extract_features(r, categories_map, len(r)-1)))
    # print(labelpoint_RDD.take(1))
    # (train_data, validation_data, test_data)
    result = labelpoint_RDD.randomSplit([8, 1, 1])
    return result


def create_model(train_data,
                 impurity_parm='entropy',
                 max_depth_parm=5,
                 max_bins_parm=5):
    model = DecisionTree.trainClassifier(train_data,
                                         numClasses=2,
                                         categoricalFeaturesInfo={},
                                         impurity=impurity_parm,
                                         maxDepth=max_depth_parm,
                                         maxBins=max_bins_parm)
    return model


def evaluate_model(model, validation_data):
    score = model.predict(validation_data.map(lambda x: x.features))
    score_and_labels = score.zip(validation_data.map(lambda x: x.label))
    metrics = BinaryClassificationMetrics(score_and_labels)
    auc = metrics.areaUnderROC
    return auc


def train_evaluation_model(train_data,
                           validation_data,
                           impurity_parm,
                           max_depth_parm,
                           max_bins_parm):
    start_time = time()
    model = create_model(train_data, impurity_parm, max_depth_parm, max_bins_parm)
    auc = evaluate_model(model, validation_data)
    print(f"AUC ==: {auc}")
    duration = time() - start_time
    print(f"The time to train was: {duration}")
    return auc, duration, impurity_parm, max_depth_parm, max_bins_parm, model


def show_chart(df, eval_parm, bar_parm, line_parm, y_min=0.5, y_max=0.7):
    ax = df[bar_parm].plot(kind='bar', title=eval_parm, figsize=(10, 6), legend=True, fontsize=12)
    ax.set_xlabel(eval_parm, fontsize=12)
    ax.set_ylim([y_min, y_max])
    ax.set_ylabel(bar_parm, fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(df[line_parm].values, linestyle='-', marker='o', linewidth=2, color='r')
    plt.show()


def eval_parameter(train_data, validation_data):
    impurity_list = ['gini', 'entropy']
    max_depth_list = [3, 5, 10, 15, 20, 25]
    max_bins_list = [3, 5, 10, 50, 100, 200]
    my_metrics = [
        train_evaluation_model(train_data, validation_data, impurity, max_depth, max_bins)
        for impurity in impurity_list
        for max_depth in max_depth_list
        for max_bins in max_bins_list
    ]
    s_metrics = sorted(my_metrics, key=lambda x: x[0], reverse=True)
    best_parameter = s_metrics[0]
    print(f"the best inpurity is:{best_parameter[2]}\n"
          f"the best max_depth is:{best_parameter[3]}\n"
          f"the best max_bins is:{best_parameter[4]}\n"
          f"the best AUC is:{best_parameter[0]}\n")
    best_auc = best_parameter[0]
    best_model = best_parameter[5]
    return best_auc, best_model


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


if __name__ == "__main__":
    s_time = time()
    create_spark_context()
    print("Reading data stage".center(60, "="))
    read_data()
    train_d, validation_d, test_d = prepare_data()
    train_d.persist(); validation_d.persist(); test_d.persist()
    print("Training and evaluation stage".center(60, "="))
    auc, model = eval_parameter(train_d, validation_d)
    print("Test stage,to judge it it over fitting".center(60, "="))
    test_data_auc = evaluate_model(model, test_d)
    print(f"best auc is:{format(auc, '.4f')}, test_data_auc is: {format(test_data_auc, '.4f')}, "
          f"they are only slightly different:{format(abs(float(auc)-float(test_data_auc)),'.4f')}")
    train_d.unpersist()
    validation_d.unpersist()
    test_d.unpersist()
    print("Prediction stage".center(60, "="))
    predict_data(model)
    dur = time() - s_time
    print(f"Ran this task cost : {format(dur, '.4f')} seconds ")





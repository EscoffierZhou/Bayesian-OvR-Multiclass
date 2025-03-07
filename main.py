from package import *
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 导入数据(根据实际情况进行修改,在线导入可能不成功)

# ONLINE:
# from ucimlrepo import fetch_ucirepo
# iris = fetch_ucirepo(id=53)
# data_x = iris.data.features.copy()
# data_y = iris.data.targets.copy()

csv_file_path = './iris.csv'
column_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
iris_df = pd.read_csv(csv_file_path, names=column_names)
feature_list = []
data_x = iris_df[['sepal length', 'sepal width', 'petal length', 'petal width']] # 选择特征列
data_y = iris_df[['class']]


# data (as pandas dataframes)
print("======================================")
print("Part0:导入数据")


print("Part1:特征分类(numeric/categorical)")
feature_lists_x = feature_lists(data_x)
feature_lists_y = feature_lists(data_y)
categorical_features_x, numerical_features_x = feature_lists_x
categorical_features_y, numerical_features_y = feature_lists_y
print("data_x 的 数值型特征:", numerical_features_x)
print("data_x 的 非数值性特征:", categorical_features_x)
print("data_y 的 数值型特征:", numerical_features_y)
print("data_y 的 非数值型特征:", categorical_features_y)

print("Part2:检查是否存在空余数据")
# 数值型:使用最常见值补充或者KNN算法进行补充
# 非数值型:使用NaN进行补充
print("======================================")
print("正在检查data_x:")
data_x_numerical = impute_numerical(data_x, numerical_features_x, method='knn')
data_x_categorical = impute_categorical(data_x, categorical_features_x)
print("======================================")
print("正在检查data_y:")
data_y_numerical = impute_numerical(data_y, numerical_features_y,method='')
data_y_categorical = impute_categorical(data_y, categorical_features_y)
print("======================================")

print("Part3:非数值型重新编码")
print("对于data_x:",end='')
with tqdm.tqdm(total=2, desc="Data-Processing") as pbar_part3:
    data_x,feature_x_map = encoder(data_x, categorical_features_x)
    if not feature_x_map:
        feature_x_map.clear()
        del feature_x_map
    else:
        print(f"存在映射{feature_x_map}")
    pbar_part3.update(1)

    print("对于data_y:",end='')
    data_y,feature_y_map = encoder(data_y, categorical_features_y)
    if not feature_y_map:
        feature_y_map.clear()
        del feature_y_map
    else:
        print(f"存在映射{feature_y_map}")
        classifier_map = feature_y_map
        converted_feature_map = {}
        for k, v in classifier_map['class'].items():
            converted_feature_map[v] = k
    pbar_part3.update(1)
print("======================================")

print("Part4:随机对应划分数据集(2:1),确定分类器个数")
with tqdm.tqdm(total=1, desc="Processing") as pbar_part4:
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=42)
    feature_y_map = list(data_y['class'].unique())
    n_classifiers = len(feature_y_map)
    print(f"分类器数量 N: {n_classifiers} (类别: {feature_y_map})")
    pbar_part4.update(1)  # Part4 完成，更新进度条
print("======================================")

print("Part5:OvR策略训练(训练多二元分类器,选取正分类器预测概率最大的)")
# 模型一:skcit-learn的朴素贝叶斯分类器
with tqdm.tqdm(total=1, desc="Processing") as pbar_part5:
    ovr_classifiers = train_GaussianNB_model(X_train, y_train, feature_y_map, converted_feature_map)
    y_pred = predict_GaussianNB(X_test, ovr_classifiers, feature_y_map) # 使用 OvR 模型进行预测
    accuracy = accuracy_score(y_test['class'], y_pred) # 计算准确率
    pbar_part5.update(1)
    print(f"GaussianNB(朴素贝叶斯)模型在测试集上的准确率: {accuracy:.4f}")
    print("--------------------------------------")
    # 模型二:自定义最小损失贝叶斯模型
    ovr_classifiers_params = train_costom_model(X_train, y_train, feature_y_map, converted_feature_map)
    y_pred_custom_bayes = ovr_predict_custom_bayes(X_test, ovr_classifiers_params, feature_y_map) # 使用自定义贝叶斯 OvR 模型进行预测
    accuracy_custom_bayes = accuracy_score(y_test['class'], y_pred_custom_bayes) # 计算准确率
    pbar_part5.update(1)
    print(f"自定义最小损失贝叶斯模型在测试集上的准确率: {accuracy_custom_bayes:.4f}")
print("======================================")
print("Part6:随机对应划分数据集(2:1),确定分类器个数")
# print("==========plot============")
# plot(y_test['class'], y_pred, 'GaussianNB')
# plot(y_test['class'], y_pred_custom_bayes, '自定义贝叶斯')
# print("======confusion_matrix====")
# cm = confusion_matrix(y_test['class'], y_pred)
# plot_cm(y_test['class'], y_pred, 'GaussianNB', feature_y_map)
# plot_cm(y_test['class'], y_pred_custom_bayes, 'Costom bayesian', feature_y_map)

print("++++++++++++++++++++++++++++")
classification_report = classification_metrics(y_test['class'], y_pred, feature_y_map)
print("朴素贝叶斯性能报告:")
print(f"总体准确率 (Accuracy): {classification_report['accuracy']:.4f}")
print(f"总体错误率 (Error Rate): {classification_report['error_rate']:.4f}")
for class_name, class_metrics in classification_report['per_class_metrics'].items():
    print("-----------------------")
    print(f"类别 '{class_name}':")
    print(f"  精确率 (Precision): {class_metrics['precision']:.4f}")
    print(f"  召回率 (Recall): {class_metrics['recall']:.4f}")
    print(f"  F1 分数 (F1-score): {class_metrics['f1_score']:.4f}")
print("++++++++++++++++++++++++++++")
classification_report = classification_metrics(y_test['class'], y_pred_custom_bayes, feature_y_map)
print("自定义贝叶斯性能报告:")
print(f"总体准确率 (Accuracy): {classification_report['accuracy']:.4f}")
print(f"总体错误率 (Error Rate): {classification_report['error_rate']:.4f}")
for class_name, class_metrics in classification_report['per_class_metrics'].items():
    print("-----------------------")
    print(f"类别 '{class_name}':")
    print(f"  精确率 (Precision): {class_metrics['precision']:.4f}")
    print(f"  召回率 (Recall): {class_metrics['recall']:.4f}")
    print(f"  F1 分数 (F1-score): {class_metrics['f1_score']:.4f}")

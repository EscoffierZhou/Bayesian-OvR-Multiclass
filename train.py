from package import *
def train_GaussianNB_model(X_train,y_train,feature_y_map,convert):
    ovr_classifiers = {} # 存储 OvR 分类器的字典
    for key in feature_y_map:
        print(f"训练类别 '{convert[key]}' 的二元分类器")
        # 筛选正负数据集
        y_train_binary = y_train['class'].apply(lambda x: 1 if x == key else 0)
        # 将目标标签转换为二元 (1: 是当前类别, 0: 不是)
        X_positive = X_train[y_train_binary == 1]
        X_negative = X_train[y_train_binary == 0]
        # 合并正负数据集
        X_train_binary = pd.concat([X_positive, X_negative])
        y_train_binary = pd.concat([y_train_binary[y_train_binary == 1], y_train_binary[y_train_binary == 0]])
        # 训练 GaussianNB 分类器
        classifier = GaussianNB()
        classifier.fit(X_train_binary, y_train_binary)
        ovr_classifiers[key] = classifier # 存储训练好的分类器
    return ovr_classifiers

def predict_GaussianNB(X_test, ovr_classifiers, feature_y_map_keys):
    y_pred = []
    y_proba_list = []
    for index, sample in X_test.iterrows(): # 遍历测试集样本
        probabilities = {} # 存储每个分类器预测的概率
        for key in feature_y_map_keys: # 遍历每个分类器
            classifier = ovr_classifiers[key]
            # 预测样本属于正类的概率 (OvR 中我们关注正类概率)
            proba = classifier.predict_proba(sample.to_frame().transpose())[0][1] # 获取正类概率
            probabilities[key] = proba # 存储概率
        predicted_class = max(probabilities, key=probabilities.get) # 选择概率最大的类别作为预测结果
        y_pred.append(predicted_class) # 添加预测类别
    return y_pred
def train_costom_model(X_train,y_train,feature_y_map,convert):
    ovr_classifiers_params = {}  # 存储 OvR 分类器参数的字典 (均值向量, 协方差矩阵)
    for key in feature_y_map:
        print(f"训练类别 '{convert[key]}' 的二元分类器")
        # 筛选正负数据集
        y_train_binary = y_train['class'].apply(lambda x: 1 if x == key else 0)
        X_train_positive = X_train[y_train_binary == 1]
        X_train_negative = X_train[y_train_binary == 0]
        # 准备自定义贝叶斯分类器所需的训练数据格式 (DataFrame 格式)
        train_positive_df = X_train_positive
        train_negative_df = X_train_negative
        # 存储训练数据, 供预测时使用 (计算后验概率时需要用到训练数据)
        ovr_classifiers_params[key] = {'train_positive': train_positive_df,
                                       'train_negative': train_negative_df}  # 存储训练数据
    return ovr_classifiers_params
def ovr_predict_custom_bayes(X_test, ovr_classifiers_params, feature_y_map_keys):
    y_pred = []
    for index, sample in X_test.iterrows(): # 遍历测试集样本
        probabilities = {} # 存储每个分类器预测的概率
        for key in feature_y_map_keys: # 遍历每个分类器 (类别)
            params = ovr_classifiers_params[key]
            train_positive_df = params['train_positive']
            train_negative_df = params['train_negative']

            # 使用自定义贝叶斯分类器计算后验概率 (getProbabilities 函数)
            prob1, prob2 = getProbabilities(sample, train_positive_df, train_negative_df) #  注意: getProbabilities 内部会处理数据转置
            # prob1 是正类 (当前类别 key) 的后验概率, prob2 是负类的后验概率
            probabilities[key] = prob1 # 存储正类概率

        predicted_class = max(probabilities, key=probabilities.get) # 选择概率最大的类别作为预测结果
        y_pred.append(predicted_class) # 添加预测类别

    return y_pred
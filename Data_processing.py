from package import KNNImputer,tqdm


# 1.形成特征列表,便于探查/补充数据
def feature_lists(df):
    categorical_features = []
    numerical_features = []
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'category':
            categorical_features.append(col) # 非数值型list
        else:
            numerical_features.append(col) # 数值型list
    return categorical_features, numerical_features

def impute_numerical(df, numerical, method):
    if not numerical: #判空
        print("该数据不存在数值型数据")
    else:
        for col in tqdm.tqdm(numerical, desc="数值型特征数据检查..."):
            if df[col].isnull().any(): # 检查该列是否有缺失值
                print(f"数值型 '{col}' 存在缺失值，正在使用 {method} 方法填充...")
                if(method == ''):
                    most_frequent_value = df[col].mode()[0]
                    df[col] = df[col].fillna(most_frequent_value)
                else:
                    imputer = KNNImputer(n_neighbors=5)
                    df[col] = imputer.fit_transform(df[[col]])
            else:
                print(f"数值型'{col}' 检查完成,无缺失值")
        return df

def impute_categorical(df, categorical):
    if not categorical:
        print("该数据不存在非数值型数据")
    else:
        for col in tqdm.tqdm(categorical, desc="非数值型特征数据检查..."):
            if df[col].isnull().any():
                df[col] = df[col].fillna('NaN')
                print(f"非数值型'{col}' 使用'NaN'填充完成")
            else:
                print(f"非数值型'{col}' 检查完成,无缺失值")
        return df

def encoder(df, categorical):
    result_map = {}
    if not categorical:
        print("该数据不存在非数值型数据，无需编码。")
        return df,result_map
    else:
        print("开始进行非数值型特征编码...")
        for col in tqdm.tqdm(categorical, desc="特征编码进度"):
            unique_values = list(df[col].unique())
            value_mapping = {val: i for i, val in enumerate(unique_values)} # 创建数值映射 (这里先用 0, 1, 2... 编码)
            df[col] = df[col].map(value_mapping)  # 使用 .map() 进行值替换
            result_map[col] = value_mapping
            print(f"非数值特征 '{col}' 编码完成。")
        return df,result_map


def ovr_predict(X_test, ovr_classifiers, feature_y_map_keys):
    y_pred = []
    for index, sample in X_test.iterrows(): # 遍历测试集样本
        probabilities = {} # 存储每个分类器预测的概率
        for key in feature_y_map_keys: # 遍历每个分类器
            classifier = ovr_classifiers[key]
            # 预测样本属于正类的概率 (OvR 中我们关注正类概率)
            proba = classifier.predict_proba(sample.to_frame().transpose())[0][1] # 获取正类概率
            probabilities[key] = proba # 存储概率

        predicted_class = max(probabilities, key=probabilities.get) # 选择概率最大的类别作为预测结果
        y_pred.append(predicted_class) # 添加预测类别
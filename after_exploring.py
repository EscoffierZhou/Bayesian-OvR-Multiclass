from package import *

def plot(y_true, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y_pred) # 使用 seaborn 的 countplot 绘制柱状图
    plt.title(f'{model_name} 模型预测类别分布')
    plt.xlabel('预测类别')
    plt.ylabel('数量')
    plt.show()

def plot_cm(y_true, y_pred, model_name, class_names):
    cm = confusion_matrix(y_true, y_pred) # 正确调用 sklearn.metrics.confusion_matrix, 只需 y_true 和 y_pred
    cm_matrix = pd.DataFrame(data=cm, columns=['AP2','AP1', 'AN0'],
                             index=['PP2','PP1', 'PN0'])
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu') # 使用 heatmap 可视化混淆矩阵
    plt.title(f'{model_name} confusion_matrix')
    plt.show()

def roc_auc_curve(y_true, y_proba, class_names):
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        fpr, tpr, thresholds = roc_curve(y_true == class_name, y_proba[class_name]) # 计算每个类别的 ROC 曲线
        auc_score = roc_auc_score(y_true == class_name, y_proba[class_name]) # 计算每个类别的 AUC 值
        plt.plot(fpr, tpr, label=f'类别 {class_name} (AUC = {auc_score:.2f})') # 绘制 ROC 曲线
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测') # 绘制随机猜测线
    plt.xlabel('FP')
    plt.ylabel('TP')
    plt.title('ROC 曲线')
    plt.legend(loc="lower right")
    plt.show()

def classification_metrics(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=class_names) # 计算混淆矩阵 (使用 sklearn.metrics.confusion_matrix)
    n_classes = len(class_names)
    metrics = {}

    # 1. 准确率 (Accuracy)
    accuracy = np.trace(cm) / np.sum(cm) # 对角线元素之和除以所有元素之和
    metrics['accuracy'] = accuracy

    # 2. 错误率 (Error Rate)
    error_rate = 1 - accuracy
    metrics['error_rate'] = error_rate

    # 3. 每个类别的 Precision, Recall, F1-score
    per_class_metrics = {}
    for i in range(n_classes):
        class_name = class_names[i]
        tp = cm[i, i] # True Positive
        fp = np.sum(cm[:, i]) - tp # False Positive (当前列，除去 TP)
        fn = np.sum(cm[i, :]) - tp # False Negative (当前行，除去 TP)

        # Precision (防止除以 0 错误)
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        # Recall (防止除以 0 错误)
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        # F1-score (只有当 Precision 和 Recall 都有意义时才计算)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        per_class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    metrics['per_class_metrics'] = per_class_metrics

    return metrics
def roc_auc_curve(y_true, y_proba, class_names):
    """
    绘制 OvR (One-vs-Rest) ROC-AUC 曲线。

    Args:
        y_true (pd.Series): 真实标签 (例如 y_test['class']).
        y_proba (pd.DataFrame): 模型预测的概率矩阵，列名为类别名称，每一列是样本属于该类别的概率.
        class_names (list): 类别名称列表 (例如 feature_y_map).
    """
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        fpr, tpr, thresholds = roc_curve(y_true == class_name, y_proba[class_name]) # 计算每个类别的 ROC 曲线
        auc_score = roc_auc_score(y_true == class_name, y_proba[class_name]) # 计算每个类别的 AUC 值
        plt.plot(fpr, tpr, label=f'类别 {class_name} (AUC = {auc_score:.2f})') # 绘制 ROC 曲线
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测') # 绘制随机猜测线
    plt.xlabel('假正率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('OvR ROC 曲线')
    plt.legend(loc="lower right")
    plt.show()
# 202318140413 周兴
import numpy as np

# 贝叶斯核心

# 1.计算似然度(P(x|ω))
def likelihood(xvec, meanvec, covmat, dims = 7):
    # xvec:特征向量(x vector)
    # meanvec:均值向量(mean vector)
    # convmat:协方差矩阵(covariance matrix)
    # dims:数据的维度（特征的数量） # 需要i修改


    # 计算数据点和均值的差值
    # 计算多元正态分布的归一化常数(保证概率密度结果为1)(通过协方差矩阵行列式)
    # 计算多元正态分布的e的指数部分:!!
    # 综合以上的总和,通过归一化常数与指数项相乘,得到似然概率
    d = xvec - meanvec
    a = 1 / (((2 * np.pi) ** (dims/2)) * (np.linalg.det(covmat)**(1/2)))
    expon = -(1/2) * np.matmul(np.transpose(d), np.matmul(np.linalg.inv(covmat), d))
    expon =  np.exp(expon)

    likelihood = a * expon
    return likelihood

# 2.计算先验概率(P(ω))
# 基于训练数据中的类别频率
def getPrior(reqClass, class1, class2):
    #for binary classifiers
    return len(reqClass[0])/(len(class1[0]) + len(class2[0]))

# 3.基于后验概率进行决策
def action(p1, p2):
    # for binary classifiers
    if(p1 >= p2):
        return 1
    else:
        return 0

# 4.计算均值向量
# 获取长度,初始化记录数组,计算该特征在所有数据点上的均值
def getmeanVec(train):
    n = len(train)
    meanvec = np.zeros(n)
    for i in range(len(train)):
        meanvec[i] = np.mean(train[i])
    return meanvec

# 5.计算后验概率(P(ω|x))(通过先验概率P(ω)乘以似然度P(x|ω))
def getProbabilities(row, t1, t2):
    #  将类别n的训练数据转换为 NumPy 数组并进行转置
    train1 = np.asarray(t1.values.tolist()).transpose()
    train2 = np.asarray(t2.values.tolist()).transpose()

    # 计算类别n训练数据的协方差矩阵
    cov1 = np.cov(train1)
    cov2 = np.cov(train2)
    # 计算类别的均值向量
    meanvec1 = getmeanVec(train1)
    meanvec2 = getmeanVec(train2)
    # 计算类别的先验概率。
    prior1 = getPrior(train1, train1, train2)
    prior2 = getPrior(train2, train1, train2)
    # 计算似然度
    p1 = likelihood(row, meanvec1, cov1)
    p2 = likelihood(row, meanvec2, cov2)
    # 计算贝叶斯的分母
    evidence = p1 * prior1 + p2 * prior2
    # 计算后验概率
    prob1 = p1 * prior1 / evidence
    prob2 = p2 * prior2 / evidence
    # prob1, prob2,包含类别1和类别2的后验概率
    return prob1, prob2

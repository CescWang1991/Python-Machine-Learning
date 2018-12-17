from sklearn import datasets
import math
import numpy as np


def createFeatList(feat):
    """
    Function:        对每一个特征，生成特征集合，集合包含不重复的特征元素

    Parameters:      feat: numpy array of shape [n_samples, n_features]
                           数据的特征

    Returns:         featSet : 特征集合
    """
    numFeat = len(feat[0])
    featSet = [list() for i in range(numFeat)]
    for i in range(numFeat):
        featSet[i] = list(set(feat[:, i]))
    return featSet


def createLabelProb(labels):
    """
    Function:        生成标签集合，记录标签值出现的先验概率

    Parameters:      labels: numpy array of shape [n_samples, 1]
                             数据的分类

    Returns:         labelCount: 记录标签值出现的次数
                     labelProb: 记录标签值出现的先验概率
    """
    labelCount = dict()
    labelProb = dict()
    m = len(labels)
    # 计算标签的先验概率时，使用Laplacian correction，P(C) = |D_c| + 1 / |D| + N
    N = len(list(set(labels)))      # N表示训练集D中的类别数
    for label in labels:
        labelCount[label] = labelCount.get(label, 0) + 1
    for label, count in labelCount.items():
        labelProb[label] = float(count + 1) / float(m + N)
    return labelCount, labelProb


def setFeatLabelPorb(feat, labels, labelCount, featSet):
    """
    Function:        计算每个特征的特征值在相应的标签下出现的后验概率。
                     对于标签值x_i，其后验概率P(x_i | C)可以通过访问featLabelProb[i][C][x_i]的value得到。
                     其中i表示特征的索引值，C为标签值，x_i为特征值。

    Parameters:      feat: numpy array of shape [n_samples, n_features]
                           数据的特征
                     labels: numpy array of shape [n_samples, 1]
                             数据的分类
                     labelCount: 见createLabelProb的返回值
                     featSet: 见createFeatList的返回值

    Returns:         featLabelCount: dict(int, dict((feat, label), count))
                                     第一层字典，key为特征的索引，value为一个字典
                                     第二层字典，key为标签值组，value为一个字典
                                     第三层字典，key为特征值，value为后验概率
    """
    featLabelProb = dict()
    for j in range(len(feat[0])):
        # 第一层字典，key为特征的索引
        featLabelProb[j] = dict()
        # n为第j个属性可能的取值数
        n = len(featSet[j])
        # 第二层字典，key为标签值组，为每一个标签值设置一个字典value
        for label in labelCount.keys():
            featLabelProb[j][label] = dict()
        # 第三层字典，key为特征值，value为后验概率
        for i in range(len(feat)):
            # m为label的出现次数
            m = labelCount[labels[i]]
            featLabelProb[j][labels[i]][feat[i][j]] = featLabelProb[j][labels[i]].get(feat[i][j], 1/(m+n)) + 1/(m+n)

    return featLabelProb


class NaiveBayes:

    def __init__(self):
        self.labelCount = None
        self.labelProb = None

    def fit(self, feat, labels):
        self.labelCount, self.labelProb = createLabelProb(labels)
        self.featSet = createFeatList(feat)
        self.featLabelProb = setFeatLabelPorb(feat, labels, self.labelCount, self.featSet)

    def predict(self, features):
        m = len(features)
        labels = [0] * m
        for j in range(len(features)):
            # 建立一个字典，用于保存每一个label在当前feat下的后验概率
            prob = {}
            for label in self.labelCount.keys():
                prob[label] = math.log(self.labelProb[label])        # 类别先验概率的对数值
                for i in range(len(features[j])):
                    m = self.labelCount[label]
                    n = len(self.featSet[i])
                    # 这里get函数的预设值为特征值不存在时的预设值
                    prob[label] += math.log(self.featLabelProb[i][label].get(features[j][i], 1/(m+n)))
            labels[j] = sorted(prob.items(), key=lambda x:x[1], reverse=True)[0][0]

        return np.array(labels)


if '__main__' == __name__:
    # 生成训练集数据, m = 1797, n = 64
    data = datasets.load_digits()
    feat = data['data']
    labels = data['target']
    clf = NaiveBayes()
    clf.fit(feat[:1700], labels[:1700])
    print(clf.predict(feat[1700:]))
    print(labels[1700:])
from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# the input vector to classify called inX                       # [x , y]
# our full matrix of training examples called dataSet           # [[x1, y1], [x2, y2], ...]
# a vector of labels called labels                              # [l1, l2 ,...]
# the number of nearest neighbors to use in the voting k
def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet             # diffMat: [x1 - x2, y1 - y2]
    sqDiffMat = diffMat ** 2                                    # 平方
    sqDistances = sqDiffMat.sum(axis=1)                         # 而当加入axis=1以后就是将一个矩阵的每一行向量相加 (x1 + y1)
    distances = sqDistances ** 0.5                              # 根号
    sortedDistIndicies = distances.argsort()                    # 排序
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]              # 按排序从小到大取label
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


group, labels = createDataSet()
inX = (2.0, 1.2)
print(classify(inX, group, labels, 3))

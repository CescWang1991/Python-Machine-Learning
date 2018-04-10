from classification.function import *
import operator


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
        voteIlabel = labels[sortedDistIndicies[i]][0]           # 按排序从小到大取label
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals


def datingClassTest(filename):
    hoRatio = 0.10
    labels, features = file2matrix(filename)
    normDataSet, ranges, minVals = autoNorm(features)
    m = normDataSet.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normDataSet[i,:],normDataSet[numTestVecs:m,:], labels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, labels[i]))
        if (classifierResult != labels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))


filename = './data/sample_multiclass_classification_data.txt'
datingClassTest(filename)
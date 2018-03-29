from classification.function import *


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:                    # Create dictionary of all possible classes
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries   # Logarithm base 2
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            prev = featVec[:axis]
            post = featVec[axis + 1:]
            retDataSet.append(hstack((prev, post)))
    return retDataSet


def chooseBestFeatureToSplit(dataSet, labels):
    numFeatures = len(dataSet[0])
    baseEntropy = calcShannonEnt(labels)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)                                #Sets are like lists, but a value can occur only once.
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet, labels)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    classLabel = -1.0
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[str(firstStr)]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def datingClassTest(filename):
    classes, features = file2matrix(filename)
    dataSet = hstack((features, classes))
    hoRatio = 0.10
    m = dataSet.shape[0]
    trainData = dataSet
    testData = dataSet[int((1-hoRatio)*m):m]
    labels = ['name', 'gender', 'height', 'weight']
    myTree = createTree(trainData, labels)
    for featVec in testData:
        labels = ['name', 'gender', 'height', 'weight']
        result = classify(myTree, labels, featVec)
        print("the classifier came back with: %d, the real answer is: %d" % (result, featVec[-1]))



filename = './data/sample_multiclass_classification_data.txt'
datingClassTest(filename)

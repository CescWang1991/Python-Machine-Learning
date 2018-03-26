from classification.function import *
import random
import numpy as np

def sigmoid(inX):
    return 1.0/(1+exp(-inX))


def gradAscent(features, labels):
    m, n = shape(features)
    alpha = 0.01
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(np.dot(features, weights))
        error = (labels - h)
        weights = weights + alpha * np.dot(features.transpose(), error)
    return weights

def stocGradAscent(features, labels):
    numIters = 150
    m, n = shape(features)
    weights = ones(n)
    for j in range(numIters):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(np.dot(features[randIndex], weights))
            error = labels[randIndex] - h
            weights = weights + alpha * error * features[randIndex]
            del(dataIndex[randIndex])
    newWeights = zeros((n, 1))
    for k in range(n):
        newWeights[k][0] = weights[k]
    return newWeights


def classify(testFeat, testLabels, weights):
    values = sigmoid(np.dot(testFeat, weights))
    predictions = zeros((len(testLabels), 1))
    errorCount = 0.0
    for i in range(len(testLabels)):
        if values[i] > 0.5:
            predictions[i] = 1
        else:
            predictions[i] = 0
        print("the classifier came back with: %d, the real answer is: %d" % (predictions[i], testLabels[i]))
        if (predictions[i] != testLabels[i]):
            errorCount += 1.0
            print("item %d is different" %i)
    print("the total error rate is: %f" % (errorCount / float(len(testLabels))))



filename = './data/sample_binary_classification_data.txt'
labels, features = file2matrix(filename)
hoRatio = 0.20
m, n = shape(features)
testFeat = features[0:int(hoRatio*m)]
trainFeat = features[int(hoRatio*m):m]
testLabels = labels[0:int(hoRatio*m)]
trainLabels = labels[int(hoRatio*m):m]
w0 = gradAscent(trainFeat, trainLabels)
w1 = stocGradAscent(trainFeat, trainLabels)
print("classify with gradient ascent")
classify(testFeat, testLabels, w0)
print("\nclassify with stochastic gradient ascent")
classify(testFeat, testLabels, w1)

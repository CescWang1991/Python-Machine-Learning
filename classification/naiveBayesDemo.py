# python 3.
# naiveBayesAlgo: simple algorithm for naive bayes, prob of each feature is according to occurs
#                 in training data.

from classification.function import *
import math


def createFeatList(features):
    numFeatures = len(features[0])
    featSet = [list() for i in range(numFeatures)]
    for i in range(numFeatures):
        featSet[i] = list(set(features[:, i]))
    return featSet


def createLabelSet(labels):
    labelSet = list(set(labels[:, 0]))
    return labelSet


def setOfFeature2Counts(features, labels):
    featSet = createFeatList(features)
    labelSet = createLabelSet(labels)
    featCounts = {}
    for i in range(len(features[0])):
        featCounts[i] = {}
        for feature in featSet[i]:
            featCounts[i][feature] = [0, 0, 0]
        for j in range(len(labels)):
            featCounts[i][features[j][i]][labelSet.index(labels[j][0])] += 1
    return featCounts


def setCondProb(inputSet, features, labels):
    numEntity = len(features[:])
    numLabel = len(createLabelSet(labels))
    featCounts = setOfFeature2Counts(features, labels)

    labelProb = {}
    for label in labelSet:
        if label not in labelProb.keys():
            labelProb[label] = 0
    for label in labels:
        labelProb[label[0]] += 1
    condProb = [0.0 for i in range(numLabel)]
    for i in range(len(inputSet)):
        try:
            counts = featCounts[i][inputSet[i]]
        except:
            counts = [1, 1, 1]
        for j in range(len(counts)):
            if counts[j] == 0:
                counts[j] = 1
            condProb[j] = condProb[j] + log(counts[j] / numEntity)
    for k in range(numLabel):
        condProb[k] = condProb[k] + log(labelProb[labelSet[k]] / numEntity)
    return condProb


labels, features = file2matrix('data\\iris_libsvm.txt')
featSet = createFeatList(features)
labelSet = createLabelSet(labels)
featProbs = setOfFeature2Counts(features, labels)
inputSet = [3.0, 2.5, 1.1, 1.5]
for i in range(len(inputSet)):
    try:
        print(featProbs[i][inputSet[i]])
    except:
        print([1, 1, 1])
condProb = setCondProb(inputSet, features, labels)
print(labelSet[condProb.index(max(condProb))])

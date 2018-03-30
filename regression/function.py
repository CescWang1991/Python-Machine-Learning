from numpy import *


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    fr = open(filename)
    numberOfColumns = len(fr.readline().split(' ')) - 1
    labels = zeros((numberOfLines, 1))
    features = zeros((numberOfLines, numberOfColumns))
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(' ')
        labels[index, 0] = listFromLine[0]
        for vector in listFromLine[1:len(listFromLine)]:
            list = vector.split(":")
            features[index, int(list[0])-1] = list[1]
        index += 1
    return labels, features


def libsvm2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    labels = zeros((numberOfLines, 1))
    features = []
    fr = open(filename)
    numberOfColumns = 0
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(' ')
        labels[index, 0] = listFromLine[0]
        index += 1
        featureLine = []
        firstFeatureIndex, firstFeature = listFromLine[1].split(':')
        lastFeatureIndex, lastFeature = listFromLine[1].split(':')
        print(firstFeatureIndex)
        for vector in listFromLine[1:len(listFromLine)]:
            list = vector.split(":")
            print(list[0])
            featureLine[int(list[0]) - 1] = list[1]
            features = np.append(features, featureLine)
    return features
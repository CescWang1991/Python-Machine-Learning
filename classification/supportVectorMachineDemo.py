from classification.function import *
import random
import numpy

# i: is the index of our first alpha.
# m: is the total number of alphas.
def selectJrand(i, m):
    j=i
    while (j==i):
        j = int(random.uniform(0, m))
    return j

# clips alpha values that are greater than H or less than L.
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels)
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            # fXi is our prediction of the class.
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            # The error Ei is based on the prediction and the real class of this instance.
            Ei = fXi - float(labelMat[i])
            # check to see that the alpha isn’t equal to 0 or C.
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # Guarantee alphas stay between 0 and C
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if (L == H):
                    # print("L==H")
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[j, :] * dataMatrix[j, :].T
                if (eta >= 0):
                    # print("eta>=0")
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    # print("j not moving enough")
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                # print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
            # print("iteration number: %d" % iter)
    return b, alphas

def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels)
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i], X[i,:].T)
    return w

def classifyTest(ws, b, dataMat, labelMat):
    m, n = shape(dataMat)
    predictions = zeros((len(labelMat), 1))
    errorCount = 0.0
    for i in range(m):
        predictions[i] = dataMat[i] * mat(ws) + b
        if(predictions[i] > 0): predictions[i] = 1.0
        else: predictions[i] = -1.0
        print("the classifier came back with: %d, the real answer is: %d" % (predictions[i], labelMat[i]))
        if(predictions[i] != labelMat[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(len(labelMat))))

filename = './data/sample_binary_classification_data.txt'
labels, features = file2matrix(filename)
for i in range(len(labels)):
    if labels[i] == 0.0:
        labels[i] = -1.0
hoRatio = 0.20
m, n = shape(features)
testFeats = features[0:int(hoRatio*m)]
trainFeats = features[int(hoRatio*m):m]
testLabels = labels[0:int(hoRatio*m)]
trainLabels = labels[int(hoRatio*m):m]
b, alphas = smoSimple(trainFeats, trainLabels, 0.6, 0.001, 40)
ws = calcWs(alphas, trainFeats, trainLabels)
classifyTest(ws, b, testFeats, testLabels)

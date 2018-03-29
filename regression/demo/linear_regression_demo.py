from numpy import *
from sklearn import linear_model
from regression.function import file2matrix


def stand_regres(features, labels):
    xMat = features
    yMat = labels
    xTx = dot(xMat.T, xMat)
    if linalg.det(xTx) == 0.0:          # test if its determinant is zero
        print("This matrix is singular, cannot do inverse")
        return
    ws = dot(linalg.inv(xTx), dot(xMat.T, yMat))
    return ws.T


def ridgeRegres(features, labels, lam=0.2):
    xMat = mat(features)
    yMat = mat(labels)
    xTx = dot(xMat.T, xMat)
    denom = xTx + eye(shape(xMat)[1]) * lam
    ws = dot(linalg.inv(denom), dot(xMat.T, yMat))
    return ws.T


path = '../data/sample_linear_regression_data.txt'
labels, features = file2matrix(path)

lr = linear_model.LinearRegression()
lr.fit(features, labels)
rr = linear_model.Ridge (alpha = 0.2)
rr.fit(features, labels)

coef_1 = stand_regres(features, labels)
coef_2 = lr.coef_
coef_3 = ridgeRegres(features, labels)
coef_4 = rr.coef_

print('coeficients of standard regression is: ')
print(coef_1)
print('coeficients of linear regression in skLearn is: ')
print(coef_2)
print('coeficients of ridge regression is: ')
print(coef_3)
print('coeficients of ridge regression in skLearn is: ')
print(coef_4)
from sklearn import linear_model
from regression.function import file2matrix

path = '../data/sample_linear_regression_data.txt'
labels, features = file2matrix(path)

lr = linear_model.LinearRegression()
lr.fit(features, labels)
coef_1 = lr.coef_
print(coef_1)

rr = linear_model.Ridge (alpha = .5)
rr.fit(features, labels)
coef_2 = rr.coef_
print(coef_2)

lr_2 = linear_model.Lasso(alpha = .1)
lr_2.fit(features, labels)
coef_3 = lr_2.coef_
print(coef_3)

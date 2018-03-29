from sklearn import tree
from classification.function import file2matrix

path = '../data/sample_multiclass_classification_data.txt'
labels, features = file2matrix(path)
test = [[1, 1, 1, 1]]

clf_1 = tree.DecisionTreeClassifier()
clf_1.fit(features, labels)
print(clf_1.predict(test))


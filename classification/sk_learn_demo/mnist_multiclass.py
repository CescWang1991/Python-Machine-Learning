from sklearn.datasets import fetch_mldata

# Load MNIST dataset
mnist = fetch_mldata("MNIST original")
X, y = mnist["data"], mnist["target"]

import numpy as np

some_digit = X[36000]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
y_train_5 = (y_train == 5)  # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier

# multiclass classification
sgd_clf = SGDClassifier(max_iter=1000, random_state=42)
sgd_clf.fit(X_train, y_train)
# print(sgd_clf.predict([some_digit]))    # [3.]
some_digit_scores = sgd_clf.decision_function([some_digit])
# print(some_digit_scores)                # return 10 scores
# print(np.argmax(some_digit_scores))     # 3
# print(sgd_clf.classes_[np.argmax(some_digit_scores)])   # 3.0

from sklearn.multiclass import OneVsOneClassifier

# creates a multiclass classifier using the OvO strategy, based on a SGDClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
# print(ovo_clf.predict([some_digit]))

from sklearn.ensemble import RandomForestClassifier

# Training a RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train)
# print(forest_clf.predict([some_digit]))

from sklearn.model_selection import cross_val_score

# Let's evaluate the SGDClassifier's accuracy using the cross_val_score() function
sgd_clf_acc = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
# print(sgd_clf_acc)      # [0.84168166 0.85414271 0.87548132]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
sgd_scale_acc = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
# print(sgd_scale_acc)      # [0.90931814 0.9080954  0.91323699]

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

# make predictions using the cross_val_predict() function, then call the confusion_matrix() function
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
# plt.matshow(conf_mx, cmap="gray")
# plt.show()

from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
print(knn_clf.predict([some_digit]))

from sklearn.metrics import f1_score

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
print(f1_score(y_train, y_train_knn_pred, average="macro"))

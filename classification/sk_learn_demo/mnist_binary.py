from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


# Load MNIST dataset
mnist = fetch_mldata("MNIST original")
X, y = mnist["data"], mnist["target"]
# print(X.shape)
# print(y.shape)

# Plot an image instance
some_digit = X[24000]
some_digit_image = some_digit.reshape(28, 28)
# plt.imshow(some_digit_image, cmap="binary", interpolation="nearest")
# plt.axis("off")
# plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
y_train_5 = (y_train == 5)  # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(max_iter=5, random_state=42)
sgd_clf.fit(X_train, y_train_5)

# Divide the dataset into 3 pieces, 2 of them be train set, and another be test set
skfolds = StratifiedKFold(n_splits=3, random_state=42)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    # print(n_correct / len(y_pred))      # 0.9673, 0.93945, 0.95535

# print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))   # [0.89835 0.9405  0.96285]
cross_acc = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


never_5_clf = Never5Classifier()
never_5_acc = cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# print(never_5_acc)      # [0.908   0.9123  0.90865]

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
# print(confusion_matrix(y_train_5, y_train_pred))    # [[53422  1157], [ 1207  4214]]

precision_score(y_train_5, y_train_pred)  # == 4214 / (4214 + 1157)
recall_score(y_train_5, y_train_pred)  # == 4344 / (4344 + 1207)
f1_score(y_train_5, y_train_pred)    # 0.7563418515398297

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])

# plot the precision recall curve
# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
# plt.show()


y_train_pred_90 = (y_scores > 100000)    # set the threshold as 100000
# print(precision_score(y_train_5, y_train_pred_90))  # 0.8186370563901589
# print(recall_score(y_train_5, y_train_pred_90))     # 0.6935989669802619

fpr, tpr, thres = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


# plot_roc_curve(fpr, tpr)
# plt.show()

# print(roc_auc_score(y_train_5, y_scores))   # 0.9618633495082931

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]  # score = probability of positive class

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend()
#plt.show()

print(roc_auc_score(y_train_5, y_scores_forest))    # 0.9927261265711859

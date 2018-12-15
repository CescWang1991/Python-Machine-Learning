import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification

class adaboost:
    def __init__(self, classifier):
        self.classifier = classifier
        self.classifiers = []
        self.alphas = []

if __name__ == "__main__":
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_classes=2,
                               n_informative=1, n_clusters_per_class=1, class_sep=0.5, random_state=10)
    # plot training data with classification
    X0 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
    X1 = np.array([X[i] for i in range(len(X)) if y[i] == 1])
    plt.plot(X0, "b*")
    plt.plot(X1, "ro")
    plt.show()
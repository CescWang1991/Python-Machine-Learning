import math
import numpy as np

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, feat, labels):
        """
        Function:        Fit the model with k nearest neighbours

        Parameters:      X : numpy array of shape [n_samples,n_features]
                             Training data
                         y : numpy array of shape [n_samples]
                             Target values. Will be cast to X's dtype if necessary

        Returns:         self : returns an instance of self.
        """
        self.feat = feat
        self.labels = labels
        self.labelSet = list(set(labels))

    def predict(self, test):
        """
        Function:        Predict using the KNN model.

        Parameters:      X : numpy array of shape = (n_samples, n_features)
                             Test Samples.

        Returns          y : array, shape = (n_samples, 1)
                             Prediction of query points.
        """
        output = np.zeros((len(test), ), dtype=np.int)
        for i in range(len(test)):
            vote = {}
            distSet = []        # 保存到每一个训练点的距离，以及训练点的label
            numEntity = len(self.feat)
            numFeat = len(self.feat[0])
            for j in range(numEntity):
                dist = 0.0
                for k in range(numFeat):
                    dist += (test[i][k] - self.feat[j][k]) * (test[i][k] - self.feat[j][k])
                distSet.append((math.sqrt(dist), self.labels[j]))
            for elem in sorted(distSet, key=lambda x:x[0])[:self.k]:
                vote[elem[1]] = vote.get(elem[1], 0) + 1
            output[i] = sorted(vote.items(), key=lambda x:x[1], reverse=True)[0][0]

        return output
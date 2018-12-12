import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification

class LDA:

    def fit(self, X, y):
        """
        Function:        Fit the model with linear discriminant analysis

        Parameters:      X : numpy array of shape [n_samples,n_features]
                             Training data
                         y : numpy array of shape [n_samples]
                             Target values. Will be cast to X's dtype if necessary

        Returns:         self : returns an instance of self.
        """
        # 将训练样本中的正例和反例提取出来
        X0 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
        X1 = np.array([X[i] for i in range(len(X)) if y[i] == 1])

        # 分别求两个类别的均值与协方差矩阵
        self.mju0 = np.mean(X0, axis=0)
        self.mju1 = np.mean(X1, axis=0)
        cov0 = np.dot((X0 - self.mju0).T, (X0 - self.mju0))
        cov1 = np.dot((X1 - self.mju1).T, (X1 - self.mju1))

        # 类内散度矩阵(within-class scatter matrix)
        Sw = cov0 + cov1
        # w = Sw^-1 * (mju0 - mju1)
        self.weight = np.dot(np.mat(Sw).I, self.mju0 - self.mju1).reshape(len(self.mju0), 1)

        return self

    def predict(self, X):
        """
        Function:        Predict using the linear model.

        Parameters:      X : {array-like, sparse matrix}, shape = (n_samples, n_features)
                             Test Samples.

        Returns          y : array, shape = (1,)
                             Prediction of query points.
        """
        y_ = np.dot(X, self.weight)
        y_predict = np.ones(y_.shape)
        y0 = np.dot(self.mju0, self.weight)
        y1 = np.dot(self.mju1, self.weight)

        for i in range(len(X)):
            y_predict[i] = 0 if abs(y_[i] - y0) < abs(y_[i] - y1) else 1

        return y_predict

    def plot(self, X, y, X_test, y_test):
        """
        Function:        对训练数据和测试数据可视化.

        Parameters:      X : numpy array of shape [n_samples,n_features]
                             Training data.
                         y : numpy array of shape [n_samples]
                             Target values. Will be cast to X's dtype if necessary.
                         X : {array-like, sparse matrix}, shape = (n_samples, n_features)
                             Test Samples.
                         y : array, shape = (1,)
                             Target values of test samples.
        """
        # 对训练集数据可视化，分类为0的数据纵坐标为0，分类为1的数据纵坐标为1
        X0 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
        X1 = np.array([X[i] for i in range(len(X)) if y[i] == 1])
        X0_new = np.dot(X0, self.weight)
        X1_new = np.dot(X1, self.weight)
        y0_new = [0 for i in range(len(X0_new))]
        y1_new = [1 for i in range(len(X1_new))]
        plt.plot(X0_new, y0_new, 'b*')
        plt.plot(X1_new, y1_new, 'ro')
        # 对测试数据可视化
        X_test0 = np.dot(np.array([X_test[i] for i in range(len(X_test)) if y_test[i] == 0]), self.weight)
        X_test1 = np.dot(np.array([X_test[i] for i in range(len(X_test)) if y_test[i] == 1]), self.weight)
        plt.plot(X_test0, [0.50 for i in range(len(X_test0))], 'b*')
        plt.plot(X_test1, [0.50 for i in range(len(X_test1))], 'ro')
        # 对测试数据及其预测值可视化
        y_predict = self.predict(X_test)
        X_predict0 = np.dot(np.array([X_test[i] for i in range(len(X_test)) if y_predict[i] == 0]), self.weight)
        X_predict1 = np.dot(np.array([X_test[i] for i in range(len(X_test)) if y_predict[i] == 1]), self.weight)
        plt.plot(X_predict0, [0.25 for i in range(len(X_predict0))], 'b*')
        plt.plot(X_predict1, [0.75 for i in range(len(X_predict1))], 'ro')

        plt.show()


if '__main__' == __name__:
    # 生成训练集数据
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_classes=2,
                               n_informative=1, n_clusters_per_class=1, class_sep=0.5, random_state=10)
    clf = LDA()
    clf.fit(X, y)
    # 生成测试集数据
    X_, y_ = make_classification(n_samples=25, n_features=2, n_redundant=0, n_classes=2,
                                 n_informative=1, n_clusters_per_class=1, class_sep=0.5, random_state=10)
    clf.plot(X, y, X_, y_)
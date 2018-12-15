from sklearn import datasets
import numpy as np
from tree.TreeNode import BinaryTreeNode as TreeNode


def splitDataSet(data, axis, value):
    """
    Function:        将dataset进行划分左右子树，左子树在axis上的值小于value，右子树则大于等于。用于feature为连续值的数据。

    Parameters:      data: numpy array of shape [n_samples, n_features + 1]
                           数据的特征+数据的类别
                     axis: 划分特征所在的列
                     value: 用于划分特征的值

    Returns:         left: 左子树的数据集
                     right: 右子树的数据集
    """
    left, right = [], []
    for featVec in data:
        if featVec[axis] < value:
            left.append(featVec)
        else:
            right.append(featVec)
    return np.array(left), np.array(right)


def chooseBestFeatureToSplit(data):
    """
    Function:        将dataset进行划分左右子树，左子树在axis上的值小于value，右子树则大于等于。用于feature为连续值的数据。
                     我们使用均方差来作为损失函数，选择左右子树均方差和最小的划分方法。

    Parameters:      data: numpy array of shape [n_samples, n_features + 1]
                           数据的特征+数据的类别

    Returns:         bestFeature: 划分特征所在的列
                     bestValue: 用于划分特征的值
    """
    numFeatures = len(data[0]) - 1
    minSumVar = float('Inf')
    bestFeature = -1
    bestValue = -float('Inf')
    for i in range(numFeatures):
        # 当前axis的特征值列表
        featList = [example[i] for example in data]
        for value in featList:
            left, right = splitDataSet(data, i, value)
            probL = len(left) / float(len(data))
            probR = len(right) / float(len(data))
            # 注意处理left或right为空时的情况，此时np.var()函数会报错
            if len(left) != 0:
                if len(right) != 0:
                    newSumVar = probL * np.var(left) + probR * np.var(right)
                else:
                    newSumVar = np.var(left)
            else:
                newSumVar = np.var(right)

            if newSumVar < minSumVar:
                minSumVar = newSumVar
                bestFeature = i
                bestValue = value
    return bestFeature, bestValue


# =============================================================
# 定义回归决策树，用于预测数据集的实数值。
# =============================================================
class RegressionDecisionTree:

    def __init__(self, maxDepth):
        self.maxDepth = maxDepth
        self.root = TreeNode()


    def fit(self, X, y):
        """
        Function:        Fit the model with classification decision tree

        Parameters:      X : numpy array of shape [n_samples,n_features]
                             Training data
                         y : numpy array of shape [n_samples]
                             Target values. Will be cast to X's dtype if necessary

        Returns:         self : returns an instance of self.
        """
        n_samples = len(y)
        # 将features和labels连接成为一个完整的matrix
        dataMatrix = np.concatenate((X, y.reshape([n_samples, 1])), axis=1)
        self.createTreeNode(self.root, dataMatrix, 1)

        return self

    def createTreeNode(self, root, data, depth):
        """
        Function:        用输入的data设置树结点，并记录当前结点的深度

        Parameters:      data: numpy array of shape [n_samples, n_features + 1]
                               数据的特征+数据的类别
                         depth: 当前树结点的深度

        Returns:         root: 返回树结点
        """
        # 当前层数等于maxDepth时，将此树结点视为叶节点，此结点将不再分裂，并且将majorClass设为value
        if self.maxDepth == depth:
            y = [example[-1] for example in data]
            root.val = np.mean(y)
            root.targetValue = y
            root.isLeaf = True
            return
        # 若当前树结点不为叶节点，我们根据划分后的信息增益最大化，选择最佳的feature和value
        root.splitFeature, root.splitValue = chooseBestFeatureToSplit(data)
        leftData, rightData = splitDataSet(data, axis=root.splitFeature, value=root.splitValue)
        root.leftNode = TreeNode()
        root.rightNode = TreeNode()
        self.createTreeNode(root.leftNode, leftData, depth + 1)
        self.createTreeNode(root.rightNode, rightData, depth + 1)


    def predict(self, X):
        """
        Function:        利用该决策树模型对数据集做出预测

        Parameters:      X : numpy array of shape [n_samples, n_features]
                             Test Samples.

        Returns          y : numpy array of shape [n_samples, 1]
                             Prediction of query points.
        """
        n_smaples = len(X)
        y = np.array([0] * n_smaples)
        root = self.root
        for i in range(n_smaples):
            featVec = X[i]
            y[i] = self.searchTree(root, featVec)

        return y

    def searchTree(self, root, feat):
        """
        Function:        在决策树模型中搜索，找到对应features所在的叶节点对应的value

        Parameters:      feat : numpy array of shape [1, n_features]
                                features of one test sample

        Returns          value: 该sample所对应叶节点的值
        """
        # 如果当前结点是叶节点，直接返回其value
        if root.isLeaf:
            return root.val
        # 如果feature在划分feature的值小于划分的value，我们将搜索左子树
        if feat[root.splitFeature] < root.splitValue:
            return self.searchTree(root.leftNode, feat)
        # 否则将搜索右子树
        else:
            return self.searchTree(root.rightNode, feat)


if '__main__' == __name__:
    X, y = datasets.make_regression(n_samples=500, n_features=4, random_state=10)
    clf = RegressionDecisionTree(3)
    clf.fit(X, y)
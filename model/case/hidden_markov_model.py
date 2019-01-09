# 运用隐马尔可夫模型解决问题

import numpy as np

class HMM:

    def __init__(self, Ann, Bnm, Pi):
        """
        :param Ann: 状态转移概率矩阵，表示在任意时刻t，若状态为i，则下一时刻状态为j的概率[n代表状态值的取值个数]
        :param Bnm: 输出观测概率矩阵，表示在任意时刻t，若状态为i，则观测值j被获取的概率[m代表观测值的取值个数]
        :param Pi: 初始状态概率，表示模型初始状态为i的概率
        """
        self.A = np.array(Ann, np.float)
        self.B = np.array(Bnm, np.float)
        self.Pi = np.array(Pi, np.float)
        self.N = self.A.shape[0]
        self.M = self.B.shape[1]

    def viterbi(self, X):
        """
        给定一段观测变量，利用维特比算法，推测出最有可能的状态变量序列
        :param X: 观测序列，表示在T时间内生成的一段可见观测变量
        :return: 状态变量序列
        """
        X = np.array(X, np.int)
        T = len(X)
        Y = np.zeros(T, np.int)

        delta = np.zeros((T, self.N), np.float)
        psi = np.zeros((T, self.N), np.int)     # psi[t, i]表示在t时刻，取到第i个状态时，是由t-1时哪一个状态推测而来
        # 根据初始状态概率，推测出t=0时的状态变量
        for i in range(self.N):
            delta[0, i] = self.Pi[i] * self.B[i, X[0]]
            psi[0, i] = 0

        # 对t=1以后的时刻进行迭代
        for t in range(1, T):
            for i in range(self.N):
                delta[t, i] = self.B[i, X[t]] * np.array([delta[t-1, j] * self.A[j, i] for j in range(self.N)]).max()
                psi[t, i] = np.array([delta[t-1, j] * self.A[j, i] for j in range(self.N)]).argmax()
        # 根据delta确定最后一个时刻的状态变量
        Y[T-1] = delta[T-1].argmax()
        # 根据psi以及最后一个时刻的状态变量进行逆推所有时刻的状态变量
        for t in reversed(range(T-1)):
            Y[t] = psi[t+1, Y[t+1]]
        return Y

    def forward(self, X):
        """
        利用前向算法产生概率矩阵，前向算法与维特比算法类似，不同在于维特比关心上一时刻的概率最大值，这里关心概率总和。
        :param X: 观测序列，表示在T时间内生成的一段可见观测变量
        :return: polambda: 模型生成观测序列X的概率
                 alpha: 概率矩阵，alpha[t,i]表示第t时刻为第i个状态变量的概率
        """
        T = len(X)
        alpha = np.zeros((T, self.N), np.float)

        for i in range(self.N):
            alpha[0, i] = self.Pi[i] * self.B[i, X[0]]

        for t in range(1, T):
            for i in range(self.N):
                sum = 0.0
                for j in range(self.N):
                    sum += alpha[t-1, j] * self.A[j, i]
                alpha[t, i] = sum * self.B[i, X[t]]
        # polambda为t=T时，观测序列X的概率P(X|λ)
        sum = 0.0
        for i in range(self.N):
            sum += alpha[T-1, i]
        polambda = sum
        return alpha, polambda

    def backward(self, X):
        """
        利用后向算法产生概率矩阵
        :param X: 观测序列，表示在T时间内生成的一段可见观测变量
        :return:
        """
        T = len(X)
        beta = np.zeros((T, self.N), np.float)
        # 初值计算，将最后时刻的概率置为1
        for i in range(self.N):
            beta[T-1, i] = 1.0

        for t in reversed(range(T-1)):
            for i in range(self.N):
                sum = 0.0
                for j in range(self.N):
                    sum += self.A[i, j] * self.B[j, X[t+1]] * beta[t+1, j]
                beta[t, i] = sum

        sum = 0.0
        for i in range(self.N):
            sum += self.Pi[i] * self.B[i, X[0]] * beta[0, i]
        polambda = sum
        return beta, polambda

    def get_param(self, X):
        """
        :param X: 观测序列，表示在T时间内生成的一段可见观测变量
        :return:
        """
        self.T = len(X)
        self.X = X

    def compute_gamma(self, alpha, beta, T):
        """
        利用前向后向概率计算在t时刻每一个状态变量的概率，即单个状态的概率。
        :param alpha: 前向概率矩阵
        :param beta:  后向概率矩阵
        :param T: 运行时间
        :return: 状态变量概率矩阵
        """
        gamma = np.zeros((T, self.N), np.float)
        for t in range(T):
            sum = 0.0
            for i in range(self.N):
                sum += alpha[t, i] * beta[t, i]
            for i in range(self.N):
                gamma[t, i] = alpha[t, i] * beta[t, i] / sum

        return gamma

    def compute_xi(self, alpha, beta, T):
        """
        利用前向后向概率计算在t时刻状态为i，t+1时刻状态为j的概率，即两个状态的联合概率。
        :param alpha: 前向概率矩阵
        :param beta:  后向概率矩阵
        :param T: 运行时间
        :return: 状态变量概率矩阵
        """
        xi = np.zeros((T - 1, self.N, self.N), np.float)
        for t in range(T-1):
            sum = 0.0
            for i in range(self.N):
                for j in range(self.N):
                    sum += alpha[t, i] * self.A[i, j] * self.B[j, self.X[t+1]] * beta[t+1, j]
            for i in range(self.N):
                for j in range(self.N):
                    xi[t, i, j] = alpha[t, i] * self.A[i, j] * self.B[j, self.X[t+1]] * beta[t+1, j] / sum

        return xi

    def baum_welch(self, X):
        """
        给定观测序列，估计参数模型，使得该模型下观测序列概率最大化
        :param X: 观测序列，表示在T时间内生成的一段可见观测变量
        :return:
        """
        self.get_param(X)
        V = [k for k in range(self.M)]

        x = 1
        delta_lambda = x + 1
        times = 0
        while delta_lambda > x:
            alpha, p1 = self.forward(X)
            beta, p2 = self.backward(X)
            gamma = self.compute_gamma(alpha, beta, self.T)
            xi = self.compute_xi(alpha, beta, self.T)

            lambda_n = [self.A.copy(), self.B.copy(), self.Pi.copy()]

            for i in range(self.N):
                for j in range(self.N):
                    # 在观测O下由状态qi转移到状态qj的期望值
                    numerator = sum(xi[t, i, j] for t in range(self.T - 1))
                    # 在观测O下由状态qi转移的期望值
                    denominator = sum(gamma[t, i] for t in range(self.T - 1))
                    self.A[i, j] = numerator / denominator

            for j in range(self.N):
                for k in range(self.M):
                    # expected number of times in state j and observing symbol Vk
                    numerator = sum(gamma[t, j] for t in range(self.T) if X[t] == V[k])
                    # 在观测O下由状态qj出现的期望值
                    denominator = sum(gamma[t, j] for t in range(self.T))
                    self.B[j, k] = numerator / denominator

            for i in range(self.N):
                self.Pi[i] = gamma[0, i]
            delta_A = map(abs, lambda_n[0] - self.A)
            delta_B = map(abs, lambda_n[1] - self.B)
            delta_Pi = map(abs, lambda_n[2] - self.Pi)
            delta_lambda = sum([sum(sum(delta_A)), sum(sum(delta_B)), sum(delta_Pi)])
            times += 1
            print(times)


if __name__ == "__main__":
    Ann = [[0.20, 0.40, 0.40],                      # 状态转移概率矩阵
           [0.35, 0.20, 0.45],
           [0.50, 0.25, 0.25]]
    Bnm = [[0.25, 0.25, 0.25, 0.25, 0.00, 0.00],    # 输出观测概率矩阵
           [0.20, 0.20, 0.20, 0.20, 0.20, 0.00],
           [0.16, 0.16, 0.16, 0.16, 0.16, 0.16]]
    Pi = [0.33, 0.33, 0.33]                         # 初始状态概率

    hmm = HMM(Ann, Bnm, Pi)
    X = [1, 2, 3]
    hmm.baum_welch(X)
# 利用协同过滤算法实现一个电影推荐的应用

import pandas as pd
import math


class collaborative_filtering:

    def __init__(self, data):
        self.data = data
        self.movies = set(self.data["movieId"])     # movies是表示所有电影id的不重复集合

    def item_based(self):
        # matrix表示电影之间的相关程度，我们用字典表示，其中key为id为(i,j)的电影(i<j)，value为相关程度。
        self.matrix = dict()
        for i in self.movies:
            for j in self.movies:
                if i < j:
                    self.matrix[(i, j)] = self.pearson_sim(i, j)

    def recommend(self, user):
        """
        为用户推荐未评分过的电影，对于电影m，我们通过matrix估算该用户的评分，最后评分最高的电影即为结果
        :param user: 用户id: int
        :return: 电影id: int
        """
        reco = dict()       # reco字典记录未被user评分的电影的估计评分
        curr = []           # 被user评分过的电影id
        data = self.data[self.data["userId"] == user]
        if data.empty:
            return None
        for mid in self.movies:
            if data[data["movieId"] == mid].empty:
                reco[mid] = 0.0
            else:
                curr.append(mid)
        for mid in reco.keys():
            sumTop = 0.0
            sumBottom = 0.0
            for vid in curr:
                key = (min(mid, vid), max(mid, vid))
                if self.matrix.get(key):
                    sumTop += self.matrix[key] * float(data[data["movieId"] == vid]["rating"])
                    sumBottom += self.matrix[key]
            reco[mid] = sumTop / sumBottom
        return max(reco, key=reco.get)

    def pearson_sim(self, m1, m2):
        """
        计算两部电影的皮尔逊相关度
        :param m1: 电影1的id, int
        :param m2: 电影2的id, int
        :return: 两者的pearson相关度, float
        """
        m1_data = self.data[self.data["movieId"] == m1]
        m2_data = self.data[self.data["movieId"] == m2]
        # 找出为m1和m2共同打分的用户
        common = []
        for key in m1_data["userId"]:
            if not m2_data[m2_data["userId"] == key].empty:
                common.append(key)
        # 用dataFrame的mean函数求取这两部电影打分的平均值
        mju1 = m1_data["rating"].mean()
        mju2 = m2_data["rating"].mean()

        sumTop = 0.0        # m1和m2的协方差
        sumLeft = 0.0       # m1的均方误差
        sumRight = 0.0      # m2的均方误差
        for user in common:
            sumTop += float(m1_data[m1_data["userId"] == user]["rating"] - mju1) * float(m2_data[m2_data["userId"] == user]["rating"] - mju2)
            sumLeft += math.pow(float(m1_data[m1_data["userId"] == user]["rating"] - mju1), 2)
            sumRight += math.pow(float(m2_data[m2_data["userId"] == user]["rating"] - mju2), 2)
        sumBottom = math.sqrt(sumLeft) * math.sqrt(sumRight)

        return sumTop / sumBottom if sumBottom != 0.0 else 0.0

if __name__ == "__main__":
    ratings = pd.read_csv("../data/ratings.csv")  # ratings包含userId，movieId，rating，time
    # print(ratings[ratings["movieId"] <= 10])
    clt = collaborative_filtering(ratings[ratings["movieId"] <= 20])   # 我们选取其中的100部电影
    clt.item_based()
    for i in range(1, 100):
        print("we recommend user ", i, " the movie ", clt.recommend(i))
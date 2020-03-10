# Spark initialization
from io import StringIO
from csv import reader

class Knn:

    KNN_pool = []
    max_pool_size = 1500

    def __init__(self , filename, sc, pool_size):

        self.init_KNN(filename,sc,pool_size)

    def getPool(self):

        return self.KNN_pool

    def init_KNN(self, file_name, sc, pool_size):

        # create an RDD
        data = sc.textFile(file_name).map(lambda x: list(reader(StringIO(x)))[0])

        # return a pool_size sample subset of an RDD as pool
        self.KNN_pool = data.takeSample(False, pool_size)

    def distance(self, v1, v2):
        '''
        Now only consider Euclidean distance
        '''
        dis = 0
        for i in range(len(v1)):
            if v1[i].isnumeric():
                dis += (float(v1[i]) - float(v2[i])) ** 2

        return dis

    def KNN(self,k, instance):

        vote_pool = []
        pool_size = len(self.KNN_pool)

        for i in range(pool_size):
            sim = self.distance(instance, self.KNN_pool[i])
            vote_pool.append((sim, self.KNN_pool[i][-1]))

        # sort the vote pool on the distance
        vote_pool = sorted(vote_pool, key=lambda tup: tup[0])

        votes_for_normal = 0
        votes_for_anomaly = 0
        for i in range(k):
            if vote_pool[i][1] == "normal":
                votes_for_normal += 1
            else:
                votes_for_anomaly += 1
        if votes_for_normal >= votes_for_anomaly:
            return "normal"
        else:
            return "anomaly"


    def simple_update_pool(self, instance):

        max_pool_size = 1500

        if len(self.KNN_pool) == 1500:

            self.KNN_pool.pop()

        self.KNN_pool.append(instance)

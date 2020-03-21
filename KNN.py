# Spark initialization
from io import StringIO
from csv import reader


def init_KNN(file_name, sc, pool_size):
    # create an RDD
    data = sc.textFile(file_name).map(lambda x: list(reader(StringIO(x)))[0])

    # return a pool_size sample subset of an RDD as pool
    return data.takeSample(False, pool_size)


def distance(v1, v2):
    '''
    Now only consider Euclidean distance
    '''
    dis = 0
    for i in range(len(v1)):
        if v1[i].isnumeric():
            dis += (float(v1[i]) - float(v2[i])) ** 2

    return dis

max_size = 300
def KNN(pool ,k, instance):

    print(pool[-1])
    vote_pool = []
    pool_size = len(pool)

    for i in range(pool_size):
        sim = distance(instance, pool[i])
        vote_pool.append((sim, pool[i][-1]))

    pool.pop()
    pool.append(instance)

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





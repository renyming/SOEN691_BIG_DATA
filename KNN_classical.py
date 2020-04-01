from pyspark import SparkContext, SparkConf
from io import StringIO
from csv import reader
import heapq


def vote(instance, train, k):
    votes = []
    for t in train:
        dist = -distance(t, instance)
        heapq.heappush(votes, (dist, t))
        if len(votes) > k:
            heapq.heappop(votes)

    cnt_normal = 0
    cnt_anomaly = 0
    for v in votes:
        if v[-1][-1] == 'normal':
            cnt_normal += 1
        else:
            cnt_anomaly += 1

    if cnt_normal > cnt_anomaly:
        return 'normal'
    else:
        return 'anomaly'


def distance(x, y):
    dist = 0.0
    for i in range(len(x) - 1):
        if i not in [1, 2, 3]:
            dist += (float(x[i]) - float(y[i])) ** 2
        elif x[i] != y[i]:
            dist += 1
    return dist


if __name__ == '__main__':
    # SparkContext.setSystemProperty('spark.executor.memory', '2500m')
    conf = SparkConf().setMaster("local[*]")
    sc = SparkContext(appName="ClassicalKNN", conf=conf)

    data = sc.textFile('./source_dir/Train_clean.csv').map(lambda x: list(reader(StringIO(x)))[0])
    n_folds = 5
    folds = data.randomSplit(weights=[1.0 / n_folds] * n_folds)
    for fold in folds:
        fold.cache()

    k_range = range(3, 11)

    with open('results.txt') as file:
        for k in k_range:
            file.write("========================================================")
            for n in range(n_folds):
                train = folds[n].collect()
                test = sc.parallelize([])
                for i in range(n_folds):
                    if i != n:
                        test = test.union(folds[i])

                pred_and_labels = test.map(lambda x: (vote(x, train, k), x[-1]))

                true_pos = pred_and_labels \
                    .filter(lambda x: x[0] == x[1] and x[0] == 'anomaly') \
                    .map(lambda x: ('True positive', 1)) \
                    .reduceByKey(lambda x, y: x + y).collect()[0][1]
                false_pos = pred_and_labels \
                    .filter(lambda x: x[0] != x[1] and x[0] == 'anomaly') \
                    .map(lambda x: ('False positive', 1)) \
                    .reduceByKey(lambda x, y: x + y).collect()[0][1]
                true_neg = pred_and_labels \
                    .filter(lambda x: x[0] == x[1] and x[0] == 'normal') \
                    .map(lambda x: ('True negative', 1)) \
                    .reduceByKey(lambda x, y: x + y).collect()[0][1]
                false_neg = pred_and_labels \
                    .filter(lambda x: x[0] != x[1] and x[0] == 'normal') \
                    .map(lambda x: ('False negative', 1)) \
                    .reduceByKey(lambda x, y: x + y).collect()[0][1]

                accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
                precision = true_pos / (true_pos + false_pos)
                recall = true_pos / (true_pos + false_neg)

                output = ""

                output += "K: {:d}".format(k)
                output += "True pos: {:d}".format(true_pos)
                output += "True neg: {:d}".format(true_neg)
                output += "False pos: {:d}".format(false_pos)
                output += "False neg: {:d}".format(false_neg)
                output += "Accuracy: {:0.5f}".format(accuracy)
                output += "Precision: {:0.5f}".format(precision)
                output += "Recall: {:0.5f}".format(recall)
                output += "F1 Measure: {:0.5f}".format((2 * precision * recall / (precision + recall)))
                output += "---------------------------------------"

                file.write(output)
                print(output)

import pyspark
import KNN as real_time_KNN
from pyspark.streaming import StreamingContext
from io import StringIO
from csv import reader
from MCNN import predict
from MCNN import init_mcnn_pool


def get_results(predictions_labels):
    # 4 conditions
    #     normal:   positive
    #     abnormal: negative
    true_pos = predictions_labels \
        .filter(lambda x: x[0] == x[1] and x[0] == 'normal') \
        .map(lambda x: ('True positive', 1)) \
        .reduceByKey(lambda x, y: x + y)
    false_pos = predictions_labels \
        .filter(lambda x: x[0] != x[1] and x[0] == 'normal') \
        .map(lambda x: ('False positive', 1)) \
        .reduceByKey(lambda x, y: x + y)
    true_neg = predictions_labels \
        .filter(lambda x: x[0] == x[1] and x[0] == 'anomaly') \
        .map(lambda x: ('True negative', 1)) \
        .reduceByKey(lambda x, y: x + y)
    false_neg = predictions_labels \
        .filter(lambda x: x[0] != x[1] and x[0] == 'anomaly') \
        .map(lambda x: ('False negative', 1)) \
        .reduceByKey(lambda x, y: x + y)

    if true_pos.count() == 0:
        true_pos = sc.parallelize(('true positive', 0))
    if false_pos.count() == 0:
        false_pos = sc.parallelize(('false positive', 0))
    if true_neg.count() == 0:
        true_neg = sc.parallelize(('true negative', 0))
    if false_neg.count() == 0:
        false_neg = sc.parallelize(('false negative', 0))

    return true_pos.union(false_pos.union(true_neg.union(false_neg)))


def saveCoord(rdd):

    rdd.foreach(lambda rec: open("myoutput.txt", "a").write(rec[0] + ":" +rec[1] + '\n'))


def saveCoord2(rdd):

    rdd.foreach(lambda rec: open("myoutput2.txt", "a").write(str(rec) + '\n'))


def MCNN_predict(rdds):

    rdds.foreach(lambda x: predict(x))


def main(ssc , pool):


    # read changed files under "input_dir" folder
    lines = ssc.textFileStream("./input_dir").map(lambda x: list(reader(StringIO(x)))[0])

    # make predictions on KNN
    # predictions_knn = lines.map(lambda x: (real_time_KNN.KNN(pool, 10, x), x[-1]))
    # knn_results = get_results(predictions_knn)
    # knn_results.pprint()

    # make predictions on MCNN
    # predictions_mcnn = lines.map(lambda x: (predict(x), x[-1]))
    # mcnn_results = get_results(predictions_mcnn)
    # mcnn_results.pprint()

    lines.foreachRDD(MCNN_predict)
    lines.pprint()


    # start StreamingContext
    ssc.start()
    ssc.awaitTermination()


if __name__ == "__main__":
    # spark initialization
    conf = pyspark.SparkConf().setMaster("local[2]")
    sc = pyspark.SparkContext(appName="PysparkStreaming", conf=conf)
    ssc = StreamingContext(sc, 1)  # Streaming will execute in each 3 seconds

    KNN_pool = real_time_KNN.init_KNN('./source_dir/Train.csv', sc, 100)
    init_mcnn_pool('./source_dir/Train.csv', sc)

    # run streaming
    main(ssc, KNN_pool)

    
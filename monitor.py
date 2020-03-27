import sys
import pyspark
import numpy as np
import pandas as pd
import KNN as Knn
from pyspark.streaming import StreamingContext
from io import StringIO
from csv import reader
from MCNN import predict
from MCNN import init_mcnn_pool

np.set_printoptions(threshold=sys.maxsize)


def get_results(predictions_labels, sc):
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


def preprocessing(path):
    headers = ["duration", "protocol_type", "service", "flag", "src_bytes",
               "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
               "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
               "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
               "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
               "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
               "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
               "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
               "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
               "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "target"]
    df = pd.read_csv(path, header=None, names=headers)
    mapping = {'normal': 0, 'anomaly': 1}
    mapping_rev = {0: 'normal', 1: 'anomaly'}
    df = df.replace({"target": mapping})
    df = df._get_numeric_data()

    # clean by iqr score
    Q1 = df.quantile(0.03)
    Q3 = df.quantile(0.97)
    IQR = Q3 - Q1   # difference between 1th and 99th percentiles
    df_clean = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    print('before', df.shape)
    print('after', df_clean.shape)

    # write the cleaned data to another file 'source_dir/Train_c.csv'
    df_clean = df_clean.replace({"target": mapping_rev})
    df_clean.to_csv('./source_dir/Train_c.csv', index=False, header=False)


def saveCoord(rdd):

    rdd.foreach(lambda rec: open("myoutput.txt", "a").write(rec[0] + ":" +rec[1] + '\n'))


def saveCoord2(rdd):

    rdd.foreach(lambda rec: open("myoutput2.txt", "a").write(str(rec) + '\n'))


def MCNN_predict(rdds):

    rdds.foreach(lambda x: predict(x))


def main(ssc, pool):

    # read on Hadoop
    # lines = ssc.textFileStream("hdfs://localhost:9000/input_dir")

    lines = ssc.textFileStream("./input_dir").map(lambda x:list(reader(StringIO(x)))[0])

    # make predictions
    #predictions_labels = lines.map(lambda x: (Knn.KNN(pool, 10, x), x[-1]))
    #predictions_labels.foreachRDD(saveCoord)
    #predictions_labels.pprint()
    #pool.pprint()

    lines.pprint()
    lines.foreachRDD(MCNN_predict)

    # start StreamingContext
    ssc.start()
    ssc.awaitTermination()


if __name__ == "__main__":
    # spark initialization
    conf = pyspark.SparkConf().setMaster("local[2]")
    sc = pyspark.SparkContext(appName="PysparkStreaming", conf=conf)
    ssc = StreamingContext(sc, 1)  # Streaming will execute in each 3 seconds

    preprocessing('./source_dir/Train.csv')
    KNN_pool = Knn.init_KNN('./source_dir/Train_c.csv', sc, 100)
    init_mcnn_pool('./source_dir/Train_c.csv', sc)

    main(ssc , KNN_pool)


import pyspark
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.utils import shuffle
from pyspark.streaming import StreamingContext
from io import StringIO
from csv import reader
from MCNN import predict
from MCNN import init_mcnn_pool


def get_results(predictions_labels, sc):
    # anomaly: positive
    # normal: negative
    true_pos = predictions_labels \
        .filter(lambda x: x[0] == x[1] and x[0] == 'anomaly') \
        .map(lambda x: ('True positive', 1)) \
        .reduceByKey(lambda x, y: x + y)
    false_pos = predictions_labels \
        .filter(lambda x: x[0] != x[1] and x[0] == 'anomaly') \
        .map(lambda x: ('False positive', 1)) \
        .reduceByKey(lambda x, y: x + y)
    true_neg = predictions_labels \
        .filter(lambda x: x[0] == x[1] and x[0] == 'normal') \
        .map(lambda x: ('True negative', 1)) \
        .reduceByKey(lambda x, y: x + y)
    false_neg = predictions_labels \
        .filter(lambda x: x[0] != x[1] and x[0] == 'normal') \
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


def data_preprocessing(path):
    headers = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
               "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
               "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
               "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
               "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
               "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
               "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
               "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
               "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
               "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "target"]
    df = pd.read_csv(path, header=None, names=headers)
    df_normal = df.loc[df['target'] == 'normal']
    df_anomaly = df.loc[df['target'] == 'anomaly']
    def plot_data(sdf, title):
        x_axis = 'src_bytes'
        y_axis = 'dst_bytes'
        fig, ax = plt.subplots(figsize=(6, 6))
        clean_data_normal = sdf.loc[sdf['target'] == 'normal']
        clean_data_anomaly = sdf.loc[sdf['target'] == 'anomaly']
        ax.scatter(clean_data_normal[x_axis], clean_data_normal[y_axis], s=2, c='red')
        ax.scatter(clean_data_anomaly[x_axis], clean_data_anomaly[y_axis], s=2, c='blue')
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        plt.title(title)
        plt.show()
    def filter_outliers(dataf):
        # calc iqr score
        Q1 = dataf._get_numeric_data().quantile(0.001)
        Q3 = dataf._get_numeric_data().quantile(0.999)
        IQR = Q3 - Q1  # difference between 0.1th and 99.9th percentiles
        # clean by iqr score
        df_class_clean = dataf[~((dataf._get_numeric_data() < (Q1 - 1.5 * IQR)) | (dataf._get_numeric_data() > (Q3 + 1.5 * IQR))).any(axis=1)]
        return df_class_clean

    # First, find the outliers in normal and anomaly and ignore them
    # plot_data(df, 'before filtering')
    df_normal_clean = filter_outliers(df_normal)
    df_anomaly_clean = filter_outliers(df_anomaly)
    df_clean = pd.concat([df_normal_clean, df_anomaly_clean], axis=0, sort=False)
    # plot_data(df_clean, 'after filtering')
    print('preprocessing:')
    print('  ', df.shape)
    print('  ', df_clean.shape)
    # Second, normalize all numeric columns
    for column in df_clean._get_numeric_data():
        mmscaler = preprocessing.MinMaxScaler()
        x_array = np.array(df_clean[column].astype(float)).reshape(-1, 1)
        df_clean[column] = mmscaler.fit_transform(x_array)
    df_norm = df_clean
    # plot_data(df_norm, 'after normalizing')
    # shuffle the data then write the clean data to 'Train_clean.csv'
    df_final = shuffle(df_norm, random_state=0)
    df_final.to_csv('./source_dir/Train_clean.csv', index=False, header=False)


def saveCoord(rdd):

    rdd.foreach(lambda rec: open("myoutput.txt", "a").write(rec[0] + ":" +rec[1] + '\n'))


def saveCoord2(rdd):

    rdd.foreach(lambda rec: open("myoutput2.txt", "a").write(str(rec) + '\n'))


def MCNN_predict(rdds):

    rdds.foreach(lambda x: predict(x))


def main(ssc):
    lines = ssc.textFileStream("./input_dir").map(lambda x:list(reader(StringIO(x)))[0])

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

    data_preprocessing('./source_dir/Train.csv')
    init_mcnn_pool('./source_dir/Train_clean.csv', sc)

    main(ssc)
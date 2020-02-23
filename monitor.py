from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from io import StringIO
from csv import reader

import KNN as real_time_KNN

def main(sc , pool):

    ssc = StreamingContext(sc, 3)   #Streaming will execute in each 3 seconds
    #lines = ssc.textFileStream("hdfs://localhost:9000/input_dir")

    lines = ssc.textFileStream("./input_dir").map(lambda x: list(reader(StringIO(x))))\
                                             .map(lambda x: x[0])


    test = lines.map(lambda  x : (real_time_KNN.KNN(pool , 50 , x) , x[41]))
    test.pprint()

    #lines.pprint()
    ssc.start()
    ssc.awaitTermination()

if __name__ == "__main__":

    sc = SparkContext(appName="PysparkStreaming")
    KNN_pool = real_time_KNN.init_KNN('./source_dir/Train.csv', sc, 200)

    main(sc , KNN_pool)

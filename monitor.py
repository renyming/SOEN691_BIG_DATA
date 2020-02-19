from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from io import StringIO
from csv import reader

Test_Instance = ['0', 'tcp', 'finger', 'S0', '0', '0', '0', '0', '0', '0', '0', '0',
                 '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '39', '12', '1',
                 '1', '0', '0', '0.31', '0.1', '0', '255', '52', '0.2', '0.04', '0',
                 '0', '1', '1', '0', '0', 'anomaly']

def main():
    sc = SparkContext(appName="PysparkStreaming")
    ssc = StreamingContext(sc, 3)   #Streaming will execute in each 3 seconds
    #lines = ssc.textFileStream("hdfs://localhost:9000/input_dir")
    lines = ssc.textFileStream("./input_dir").map(lambda x: list(reader(StringIO(x))))\
                                             .map(lambda x: x[0])

    lines.pprint()
    ssc.start()
    ssc.awaitTermination()


def calculate_distance(v1 , v2):



    return 0




if __name__ == "__main__":
    main()
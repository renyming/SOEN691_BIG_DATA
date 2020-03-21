#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

r"""
 Counts words in UTF8 encoded, '\n' delimited text received from the network every second.
 Usage: network_wordcount.py <hostname> <port>
   <hostname> and <port> describe the TCP server that Spark Streaming would connect to receive data.
 To run this on your local machine, you need to first run a Netcat server
    `$ nc -lk 9999`
 and then run the example
    `$ bin/spark-submit examples/src/main/python/streaming/network_wordcount.py localhost 9999`
"""
from __future__ import print_function

import sys

from pyspark import SparkContext
from pyspark.streaming import StreamingContext

import os, shutil
from io import StringIO
from csv import reader

counter = 1
folder = "./counter"


def clean():

    for filename in os.listdir(folder):

        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))



if __name__ == "__main__":
    clean()

    # if len(sys.argv) != 3:
    #     print("Usage: network_wordcount.py <hostname> <port>", file=sys.stderr)
    #     sys.exit(-1)

    sc = SparkContext(appName="PythonStreamingNetworkWordCount")
    ssc = StreamingContext(sc, 3)

    # lines = ssc.socketTextStream(sys.argv[1], int(sys.argv[2]))
    # counts = lines.flatMap(lambda line: line.split(" ")) \
    #               .map(lambda word: (word, 1)) \
    #               .reduceByKey(lambda a, b: a+b)
    # counts.pprint()

    counter = ssc.textFileStream(folder).map(lambda x: list(reader(StringIO(x)))[0])
    counter.pprint()


    ssc.start()
    ssc.awaitTermination()
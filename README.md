<h1>Abstract</h1>
Real time network intrusion detection system is a system used to detect anomalous network activities based on streams of network traffic data. It is more flexible and scalable than signature-based intrusion detection system. In this project, we will simulate network traffic streams by replaying pre-captured network packets feature data at a certain rate. Micro-Cluster Nearest Neighbor (MC-NN) data stream classifier will be used to classify the packet as normal or anomalous traffic. The packet feature data set is labeled, and the detection result will be evaluated against the labels. In addition, MC-NN classifier will be implemented as it is not part of Spark official library. Also, comparative study will be performed between MC-NN and KNN.

<h1>Introduction</h1>
<h2>Context</h2>
Anomaly detection is the identification of the rare data instances amongst the majority of the normal data instances. The applications of using the anomaly detection including bank fraud, medical disease detection and network intrusion detection. Real time based anomaly detection over data streams requires the ability to deal with high velocity of data, and dynamically captures the constantly evolving signatures of anomaly data. Real time anomaly detection systems provide better flexibility and scalability than a signature-based systems. 
<h2>Objectives</h2>
In this project, we are going to develop a simple network monitoring application using real time classifier of data streaming to detect network anomalous traffic. Also, the performances of different real time classifiers will be compared. 
<h2>Presentation of the Problem</h2>
Network packets are generated at a massive speed on the network. Without looking at the actual payload of those packets, there are some features of the packet that can be used by the classifier, such as protocol type, service, duration and host details etc. All of those features are numerical values or textual categories. However, those features vary between packets. It’s difficult to capture such a variety of signatures by fixed rules. Thus a real time classifier needs to be used in this case to dynamically identify if a new coming packet is normal or anomalous. Anomalous packets will be dropped to protect the network from suspected intrusion activities. 
<h2>Related Work</h2>

* Implementation of MC-NN

  MC-NN is not part of the Spark library. Thus in this project we will implement MC-NN classifier.
  
* Implementation of real time KNN

  The KNN in Spark is not designed for real time classification. Time window will be added to KNN to enable real time classification ability.
  
* Comparison between MC-NN and real time KNN

  The performance of MC-CNN will be compared with that of real time KNN.
 
<h1>Materials and Methods</h1>

<h2>Dataset</h2>

The dataset was created in a military network environment.They created a LAN network which is typical in the US Air Force and they attacked in multiple ways and collected the TCP/IP dump. Each connection is a TCP packet which has a starting time ,ending time for Source IP address and target IP address. They have used three protocols TCP ,UDP and ICMP. The connection is labeled as normal or abnormal.

The dataset has 25192 rows and 42 columns. The dataset is clean. We don’t have any NULL or EMPTY values. It has 11743 rows which are classified as anomaly and the rest 13449 classified as normal. We think we will mostly use 20 columns as the other columns have zero values. We are planning on combining different columns and see which combination gives better results.Based on the initial assessment  the columns we think  are important for the classification are class, srv_error_rate, error_rate, dst_host_error_rate and dst_host_srv_rerror_rate. We are a little bit unsure about the meaning of some of the columns in the dataset. We are researching these columns to get more information.

<h2>Technologies</h2>

The main issue about our application is to find a proper data streaming source to imitate the real network environment. Here we use spark streaming library to create the file stream on the dataset. The main purpose of this project is to use different classifiers (KNN and MC-NN) to detect the network anomalies. In this case, we need consistent stream of data to test on the performance of different classifiers. Also spark operations on the streaming will be used to implement the algorithms. The detailed documentation of spark streaming can be found. 

If time is sufficient, we will try out different streaming source like kafka or Hadoop Distributed File System.

(reference: https://spark.apache.org/docs/latest/streaming-programming-guide.html)

<h2>Algorithms Description</h2>

<h3>1. K Nearest Neighbors (KNN)</h3>

KNN is a supervised machine learning model used for classification and regression. It is widely used in data analytics. KNN algorithm hinges on the assumption where similar samples locate in close proximity in the feature space. The model picks the K nearest samples in the feature space then predicts the new samples based on those K samples. In real-time data streaming, training data and testing data are continuously changing throughout data streams. 

Here’s the KNN algorithm pseudo-code:
```
for each testing instance
	find the K most nearest instance of the training set according to a distance metric
	resulting class = most frequent class label of the K nearest instances
```

(reference: https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761) 

(reference: Lecture_slides: Supervised Classification)

<h3>2. Micro-Cluster Nearest Neighbour (MC-NN)</h3>

MC-NN is a data stream classifier. It is used to handle data streams and adapts to concept drifts.

Its basic idea is to calculate the Euclidean distance between a new data instance to each micro-cluster centroid, then assign the instance to the nearest micro-cluster. If it is correctly classified, then add the instance to the micro-cluster. If misclassified, then first add the instance to the correct micro-cluster, and then increment the error counters both on the nearest micro-cluster and the correct micro-cluster, once one of the micro-clusters’ error counter exceed a predefined threshold, we split the micro-cluster into 2.

(reference: https://www.sciencedirect.com/science/article/pii/S0167739X17304685)

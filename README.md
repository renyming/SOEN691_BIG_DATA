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

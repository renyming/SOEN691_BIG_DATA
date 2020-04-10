# Abstract

As the amount of Data which is received by the Network devices goes beyond the memory constrainsts of Standard monitoring applications using Data Streaming algorithm seems like good stratergy (Cite ).For applications like Detection of Anamoly the system needs to respond quickly. In this project we stimulated the network the network traffic and implemented Data Streaming algorithm MC -NN to classify the network packets as normal or anamoly.We also implemented the KNN (offline) to compare the performance of MC-NN .In the data preperation part we used outlier detector method “IQR ” to elimate the outliers.We compared and analyzed the results of these two algorithms.
# I. Introduction

## Context

As the number of devices which are connected to network increases it leads to opensess, diversity and sharing of knowledge but it also has the scope of many security risks to the network (Cite this).
Intrusion detecton system is security application which detects attacks and intrusion behavioir  Yin et al. As the types of attacks are becoming complex we need to find creative ways to enhance intrusion detection systems Keegan et al. (2016).
Anomaly detection is the identification of the rare data instances amongst the majority of the normal data instance
//TODO 
Give context for Data Streaming Algorithms


## Objectives

* In this project, we are going to develop a simple network monitoring application using real time classifier of data streaming to detect network anomalous traffic. Also, the performances of different real time classifiers will be compared. 

## Presentation of the Problem

Network packets are generated at a massive speed on the network. Without looking at the actual payload of those packets, there are some features of the packet that can be used by the classifier, such as protocol type, service, duration and host details etc. All of those features are numerical values or textual categories. However, those features vary between packets. It’s difficult to capture such a variety of signatures by fixed rules. Thus a real time classifier needs to be used in this case to dynamically identify if a new coming packet is normal or anomalous. Anomalous packets will be dropped to protect the network from suspected intrusion activities. 
## Related Work

* N. C. N. Chu, A. Williams, R. Alhajj and K. Barker, "Data stream mining architecture for network intrusion detection," Proceedings of the 2004 IEEE International Conference on Information Reuse and Integration, 2004. IRI 2004., Las Vegas, NV, 2004, pp. 363-368.

* Yin, C., Xia, L., Zhang, S. et al. Improved clustering algorithm based on high-speed network data stream. Soft Comput 22, 4185–4195 (2018). https://doi.org/10.1007/s00500-017-2708-2
* Silva, J. A., Faria, E. R., Barros, R. C., Hruschka, E. R., de Carvalho, A. C. P. L. F., and Gama, J. 2013. Data
stream clustering: A survey. ACM Comput. Surv. 46, 1, Article 13 (October 2013), 31 pages.
DOI: http://dx.doi.org/10.1145/2522968.2522981
  
* Sang-Hyun Oh, Jin-Suk Kang, Yung-Cheol Byun, Gyung-Leen Park and Sang-Yong Byun, "Intrusion detection based on clustering a data stream," Third ACIS Int'l Conference on Software Engineering Research, Management and Applications (SERA'05), Mount Pleasant, MI, USA, 2005, pp. 220-227.

# II. Materials and Methods

## Dataset

The dataset was created in a military network environment.They created a LAN network which is typical in the US Air Force and they attacked in multiple ways and collected the TCP/IP dump. Each connection is a TCP packet which has a starting time ,ending time for Source IP address and target IP address. They have used three protocols TCP ,UDP and ICMP. The connection is labeled as normal or abnormal.

The dataset has 25192 rows and 42 columns. The dataset is clean. We don’t have any NULL or EMPTY values. It has 11743 rows which are classified as anomaly and the rest 13449 classified as normal. We think we will mostly use 20 columns as the other columns have zero values. We are planning on combining different columns and see which combination gives better results.Based on the initial assessment  the columns we think  are important for the classification are class, srv_error_rate, error_rate, dst_host_error_rate and dst_host_srv_rerror_rate. We are a little bit unsure about the meaning of some of the columns in the dataset. We are researching these columns to get more information.

(reference: https://www.kaggle.com/sampadab17/network-intrusion-detection)

## Technologies and Algorithms

Spark was used to implement kNN and MC-NN algorithms and simulate streaming data in this project.

### 1.  k Nearest Neighbours (kNN)

kNN is a supervised machine learning model used for classification and regression. It is widely used in data analytics. kNN algorithm hinges on the assumption that similar samples locate in close proximity in the feature space. The model picks the k nearest samples in the feature space then predicts the new samples based on the majority vote of those k samples.

Here’s the pseudo-code of kNN algorithm:

```
for each testing instance
	Find the k nearest instances of the training set according to a distance metric
	Resulting class = most frequent class label of the k nearest instances
```

(reference: https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761) 

(reference: Lecture_slides: Supervised Classification)

In order to provide the baseline for MC-NN performance comparison, kNN was applied to the entire dataset offline. Since there is one hyperparameter k (the number of nearest neighbours) in kNN, cross-validation was used to search for the optimized value of k. The whole dataset was randomly split into 5 folds. 1/5 was used as training, which is around 5,000 data samples, 4/5 were used for test. 

The range of k been search is [3, 9] with the increment of 1, [10, 100] with the increment of 10. 

As for the distance calculation, squared difference was used for numerical features. Categorical feature distance was considered as 1 for different category values. 

kNN is very expensive to compute, since it has to calculate the distance between one testing instance with every sample (there are more than 5,000 samples in this case). It is infeasible to run the spark job with single node. But thanks to Dr. Tristan Glatard’s sponsorship, we were able to perform the kNN evaluation on [Compute Canada](https://www.computecanada.ca/)'s cluster with a reasonable running time.

### 2. Micro-Cluster Nearest Neighbour (MC-NN)

~~(reference: https://spark.apache.org/docs/latest/streaming-programming-guide.html)

Micro Clusters Nearest Neighbour (MC-NN) is a data stream classifier. Data stream by its definition may contain infinite data instances so that MC-NN is applied to the data stream classification for the sake of its fast calculation and update on information.  

The major idea of MC-NN is to calculate the Euclidean distance between a new data instance to each micro-cluster centroid, then assign the instance to the nearest micro-cluster. If it is correctly classified, then add the instance to the micro-cluster. If misclassified, then first add the instance to the correct micro-cluster, and then increment the error counters both on the nearest micro-cluster and the correct micro-cluster, once one of the micro-clusters’ error counter exceed a predefined threshold, we split the micro-cluster into 2.

Here’s the pseudo-code of MC-NN classifier:

```
foreach Micro-Cluster in LocalSet do:
  Evalate Micro-Cluster against NewInstance;
end
Sort EvaluationDistances();
if Nearest Micro-Cluster is of the Training Items Class Label then:
   CorrectClassification Event:
   NewInstance is added into Nearest Micro-Cluster
   Nearest Micro-Cluster Error count(ϵ) reduced
else
   MisClassification Event:
   Two Micro-Clusters Identified:
   1) MC that should have been identified as the Nearest to the New Instance of the
      Classification Label.
   2) MC that incorrectly was the Nearest the New Instance.
      Training item added to the MC of the Correct Classification Label. Both identified
      Micro-Cluster have internal Error count(ϵ) incremented
   foreach Micro-Cluster Identified do:
           if MC's Error count(ϵ) exceeds Error Threshold(θ) then:
              Sub-Divide Micro-Cluster upon attribute of lagest Variance
           end
   end
end
```
(reference: https://www.sciencedirect.com/science/article/pii/S0167739X17304685)

Implementation details: 


(Remember to mention the difference of distance calculation with kNN)

# III. Results

## 1. kNN

For each k, the averaged accuracy, precision, recall and F1-score were calculated from 5 iterations, and plotted as below:

![](./report_pics/kNN_results.png)

In general, kNN obtained very good results on the dataset, all metrics are above 0.94. However, it runs very slowly. The running time for one iteration was around 6 minutes even using 32 CPU cores. 

As shown in the plot, accuracy, recall and F1-score all decrease as k increases. However, precision first shows a fluctuant decrease and then bounces back when k=10. The best overall performance appears when K=3, where the accuracy, precision and F1-score are the highest and recall is the third highest. A detailed results table is shown as below:

| k    | Accuracy | Precision | Recall  | F1-score |
| ---- | :------: | :-------: | :-----: | :------: |
| 3    | 0.98809  |  0.98811  | 0.98634 | 0.98722  |
| 4    | 0.98639  |  0.98178  | 0.98917 | 0.98546  |
| 5    | 0.98652  |  0.98632  | 0.98475 | 0.98553  |
| 6    | 0.98498  |  0.98118  | 0.98672 | 0.98393  |
| 7    | 0.98425  |  0.98457  | 0.98161 | 0.98308  |
| 8    | 0.98285  |  0.98019  | 0.98310 | 0.98164  |
| 9    | 0.98287  |  0.98331  | 0.97990 | 0.98160  |
| 10   | 0.98144  |  0.97951  | 0.98071 | 0.98010  |
| 20   | 0.97828  |  0.97972  | 0.97358 | 0.97663  |
| 30   | 0.97576  |  0.97930  | 0.96849 | 0.97386  |
| 40   | 0.97368  |  0.98071  | 0.96252 | 0.97150  |
| 50   | 0.97277  |  0.98350  | 0.95769 | 0.97041  |
| 60   | 0.97240  |  0.98400  | 0.95640 | 0.96998  |
| 70   | 0.97151  |  0.98551  | 0.95290 | 0.96893  |
| 80   | 0.97093  |  0.98593  | 0.95123 | 0.96827  |
| 90   | 0.97016  |  0.98618  | 0.94930 | 0.96739  |
| 100  | 0.96967  |  0.98625  | 0.94817 | 0.96683  |


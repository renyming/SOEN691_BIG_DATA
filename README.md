# Abstract

Real time network intrusion detection system is a system used to detect 
anomalous network activities based on streams of network traffic data. 
It is more flexible and scalable than signature-based intrusion detection 
system. In this project, we will simulate network traffic streams by replaying 
pre-captured network packets feature data at a certain rate. Micro-Cluster 
Nearest Neighbour (MC-NN) data stream classifier will be used to classify the 
packet as normal or anomalous traffic. The packet feature data set is labeled, 
and the detection result will be evaluated against the labels. In addition, 
MC-NN classifier will be implemented as it is not part of Spark official 
library. Also, comparative study will be performed between MC-NN and kNN.


# I. Introduction


## Context
Anomaly detection is the identification of the rare data instances amongst the 
majority of the normal data instances. The applications of using the anomaly 
detection including bank fraud, medical disease detection and network 
intrusion detection. Real time based anomaly detection over data streams 
requires the ability to deal with high velocity of data, and dynamically 
captures the constantly evolving signatures of anomaly data. Real time anomaly 
detection systems provide better flexibility and scalability than a 
signature-based systems. 

## Objectives
In this project, we are going to develop a simple network monitoring 
application using real time classifier of data streaming to detect network 
anomalous traffic. Also, the performances of different real time classifiers 
will be compared. 

## Presentation of the Problem
Network packets are generated at a massive speed on the network. Without 
looking at the actual payload of those packets, there are some features of the
packet that can be used by the classifier, such as protocol type, service, 
duration and host details etc. All of those features are numerical values or 
textual categories. However, those features vary between packets. 
It’s difficult to capture such a variety of signatures by fixed rules. 
Thus a real time classifier needs to be used in this case to dynamically 
identify if a new coming packet is normal or anomalous. Anomalous packets will 
be dropped to protect the network from suspected intrusion activities. 

## Related Work
* Implementation of MC-NN

    MC-NN is not part of the Spark library. Thus in this project we will 
    implement MC-NN classifier.
  
* Implementation of real time kNN

    The kNN in Spark is not designed for real time classification. 
    Time window will be added to kNN to enable real time classification 
    ability.
  
* Comparison between MC-NN and real time kNN

    The performance of MC-CNN will be compared with that of real time kNN.


# II. Materials and Methods


## Dataset

### 1. Background
The dataset consists of a variety of network intrusion simulated data in a 
military network environment. It contains 25192 rows of TCP/IP connection 
data. Each connection is a TCP packet which contains 42 columns. There is no 
NULL or EMPTY values in original dataset, while it does have outliers which we 
will eliminate during data preprocessing.

### 2. Dataset analysis
In the original dataset, features are described as below:

|     Type     |  Number  |   Columns  |
| ------------ | -------- | -----------|
| qualitative  |    4     | protocol, service, flag, target |
| quantitative |   38     | duration, land, urgent, ... |

#### (1) Class label:

In the original dataset:

|           |  Target  |   Rows  |
| --------- | -------- | ------- |
| positive  | Anomaly  |  11743  |
| negative  | Normal   |  13449  |
| total     |          |  25912  |

In the preprocessed dataset:

|           |  Target  |   Rows  |
| --------- | -------- | ------- |
| positive  | Anomaly  |  11708  |
| negative  | Normal   |  13404  |
| total     |          |  25112  |

#### (2) Correlations:

We did some research on the original dataset and the processed dataset after 
data preprocessing, and plot the correlations between features, here is the 
results:

|     dataset      | Feature Correlations                                    |     
| ---------------- | ------------------------------------------------------- | 
| Original dataset | <img src="report_pics/correlations-train.png">          | 
| Preprocessed dataset | <img src="report_pics/correlations-train-clean.png">|
<i>(the plots were generated by the code in “dataset_analysis.py”, references 
are mentioned inside the file.) </i> <br/>
<i>(the results not only show the correlations between features, 
but also ensures that preprocessing step does not change the relations.) </i>

### 3. Data preprocessing
#### (1) Outlier detection and filtering:
We used an outlier detection method which is called “IQR method” developed by 
John Tukey only on the 38 quantitative type columns. The IQR method and a result 
example is briefly described below.

* IQR method brief intro:

IQR indicates interquartile range. It is a measure of the dispersion similar 
to 2 different quantiles of the data. Tukey who developed this method consider
any data point that fall outside of either 1.5 times the IQR below the first 
quartile or 1.5 times the IQR above the third quartile to be “outliers”. 

During our preprocessing step, we consider data where 1.5 times its IQR score 
is below the <img src="https://render.githubusercontent.com/render/math?math=0.1^{th}"> 
quantile or 1.5 times its IQR score is above 
<img src="https://render.githubusercontent.com/render/math?math=99.9^{th}"> quantile 
is outlier.

(reference: http://colingorrie.github.io/outlier-detection.html)

* Results:

We take an example as a result shown below, where the horizontal feature 
indicates “src_bytes” meaning how many bytes does the source send, and the 
vertical feature indicates “dst_bytes” meaning how many bytes does the 
destination receives.

|          before filtering           |            after filtering           | 
| ----------------------------------- | ------------------------------------ |
| ![before](./report_pics/before-filtering.png) | ![after](./report_pics/filtering.png) |

* Advantages of outlier filtering:

During our experiments, outlier detection helps MC-NN model a lot. Before filtering, 
the best MC-NN model (theta=2, mx=25) performed 82.959% on F1-score, while after 
filtering, the best model (theta=2, mx=25) performed 89.272% on F1-score.

The behind reason we analyzed is: outlier detection and filtering avoid the model 
to split a parent micro cluster to accommodate extrema in the dataset. 
So micro cluster only splits when necessary, which increases the overall accuracy.

#### (2) Feature normalization:
We transformed values on the 38 quantitative columns to the range between 0 and 1, 
so that the model won’t be biased towards any of one feature while running.

We show an example result below on the “duration” feature.

|         before normalizing          |           after normalizing          | 
| ----------------------------------- | ------------------------------------ |
| ![before](./report_pics/before-norm.png) | ![after](./report_pics/normalize.png) |


### 4. Dataset Source
https://www.kaggle.com/sampadab17/network-intrusion-detection


## Technologies and Algorithms

Spark was used to implement kNN and MC-NN algorithms and simulate streaming 
data in this project.

### 1.  k Nearest Neighbours (kNN)

kNN is a supervised machine learning model used for classification and 
regression. It is widely used in data analytics. kNN algorithm hinges on the 
assumption that similar samples locate in close proximity in the feature space. 
The model picks the k nearest samples in the feature space then predicts the 
new samples based on the majority vote of those k samples.

Here’s the pseudo-code of kNN algorithm:

```
for each testing instance
	Find the k nearest instances of the training set according to a distance metric
	Resulting class = most frequent class label of the k nearest instances
```

(reference: https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761) 

(reference: Lecture_slides: Supervised Classification)

In order to provide the baseline for MC-NN performance comparison, kNN was 
applied to the entire dataset offline. Since there is one hyperparameter k 
(the number of nearest neighbours) in kNN, cross-validation was used to search 
for the optimized value of k. The whole dataset was randomly split into 5 folds. 
1/5 was used as training, which is around 5,000 data samples, 4/5 were used 
for test. 

The range of k been search is [3, 9] with the increment of 1, [10, 100] with 
the increment of 10. 

As for the distance calculation, squared difference was used for numerical 
features. Categorical feature distance was considered as 1 for different 
category values. 

kNN is very expensive to compute, since it has to calculate the distance 
between one testing instance with every sample (there are more than 5,000 
samples in this case). It is infeasible to run the spark job with single node. 
But thanks to Dr. Tristan Glatard’s sponsorship, we were able to perform the 
kNN evaluation on [Compute Canada](https://www.computecanada.ca/)'s cluster 
with a reasonable running time.

### 2. Micro-Cluster Nearest Neighbour (MC-NN)

(Needs update, kNN is offline now, only need to state the case for MC-NN)
 ~~The main issue about our application is to find a proper data streaming 
 source to imitate the real network environment. Here we used spark streaming 
 library to create the file stream on the dataset. The main purpose of this 
 project is to use different classifiers (kNN and MC-NN) to detect the network 
 anomalies. In this case, we need consistent stream of data to test on the 
 performance of different classifiers. Also spark operations on the streaming 
 will be used to implement the algorithms. The detailed documentation of spark 
 streaming can be found.~~ 

~~If time is sufficient, we will try out different streaming source like kafka 
or Hadoop Distributed File System.~~

~~(reference: https://spark.apache.org/docs/latest/streaming-programming-guide.html)

MC-NN is a data stream classifier. It is used to handle data streams and adapts 
to concept drifts.

Its basic idea is to calculate the Euclidean distance between a new data instance 
to each micro-cluster centroid, then assign the instance to the nearest micro-cluster. 
If it is correctly classified, then add the instance to the micro-cluster. 
If misclassified, then first add the instance to the correct micro-cluster, 
and then increment the error counters both on the nearest micro-cluster and the 
correct micro-cluster, once one of the micro-clusters’ error counter exceed a 
predefined threshold, we split the micro-cluster into 2.

(reference: https://www.sciencedirect.com/science/article/pii/S0167739X17304685)

(Remember to mention the difference of distance calculation with kNN)

# III. Results

## 1. kNN

For each k, the averaged accuracy, precision, recall and F1-score were calculated 
from 5 iterations, and plotted as below:

![](./report_pics/kNN_results.png)

In general, kNN obtained very good results on the dataset, all metrics are 
above 0.94. However, it runs very slowly. The running time for one iteration 
was around 6 minutes even using 32 CPU cores. 

As shown in the plot, accuracy, recall and F1-score all decrease as k increases. 
However, precision first shows a fluctuant decrease and then bounces back when k=10. 
The best overall performance appears when K=3, where the accuracy, precision and 
F1-score are the highest and recall is the third highest. A detailed results 
table is shown as below:

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


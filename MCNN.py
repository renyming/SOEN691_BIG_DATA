import os
import random
import csv
import shutil
import numpy as np
from io import StringIO
from os.path import join
from csv import reader
from mcnnmc import MC


'''
改动的地方:
1. 每个mc文件头的信息：
    1) epsilon: error counter
    2) n: number of instances
    3) centroid: centroid
2. 每个rdd需要读：epsilon, n, centroid
    1) 根据centroid算距离
    2) 根据centroid, n, instance更新centroid and n
    3) 需不需要存每个点的数据 (???)
    4) Split (???)
    5) 如果用foreachRDD，需要把(prediction, label, time) 存下来
    6) 最后等所有(prediction, label, time)都存下来后，再写另一个py文件做evaluation


'''
mc_folder = './mcnn_mcs'


# helper function
def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

# helper function
def clean_mc_folder():
    for filename in os.listdir(mc_folder):
        file_path = join(mc_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

class MC_NN:

    def __init__(self, theta):
        self.theta = theta
        self.pool = []  # a list of MC objects

        self.read_mcnn_pool()

    def read_mcnn_pool(self):
        '''
        read all centroid files into a self.pool
        :return:
        '''
        mc_files = [f for f in os.listdir(mc_folder) if f.endswith('.csv')]

        for file in mc_files:
            mc = MC(self.theta)
            if file.__contains__('normal'):
                mc.cl = 'normal'
            elif file.__contains__('anomaly'):
                mc.cl = 'anomaly'

            with open(join(mc_folder, file), 'r') as f:
                lines = f.read().splitlines()

                # the first line is epsilon (for convention)
                mc.epsilon = int(lines[0])
                mc.n = int(lines[1]) #count
                mc.centroid = [float(x) for x in lines[2].split(',') if is_number(x)] #centroids only consider the numeric number now

                # the following elements are instances in the mc
                #mc.cf_all = [[x for x in line.split(',')] for line in lines[1:]]

                # note: x.isnumeric returns False for float numbers
                #mc.cf = [[float(x) for x in line.split(',') if is_number(x)] for line in lines[1:]]

            #cf_temp = np.array(mc.cf)
            #cf_temp = np.array(mc.centroid)

            #mc.cf1_x = cf_temp * mc.n #The sum of the feature is the avarage centroid * n
            #mc.cf2_x = np.multiply(mc.cf1_x, mc.cf1_x)
            #mc.n = cf_temp.shape[0]
            #mc.centroid = mc.cf1_x / mc.n

            self.pool.append(mc)

    def euclidean_distance(self, v1, v2):
        distance = 0
        for i in range(len(v1)):
            distance += (v1[i] - v2[i]) ** 2
        return distance

    def find_true_nearest_mc(self, instance):
        true_label = instance[-1]
        features = [float(attr) for attr in instance if is_number(attr)]

        min_distance = float('inf')
        min_true_mc = None
        for mc in self.pool:
            if mc.cl == true_label:
                distance = self.euclidean_distance(features, mc.centroid)
                if distance < min_distance:
                    min_distance = distance
                    min_true_mc = mc
        return min_true_mc

    def predict_and_update_mcs(self, instance, true_label):
        # predict
        features = [float(attr) for attr in instance if is_number(attr)]
        features_np = np.array(features)

        min_distance = float('inf')
        min_mc = None
        for mc in self.pool:
            distance = self.euclidean_distance(features, mc.centroid)
            if distance < min_distance:
                min_distance = distance
                min_mc = mc

        prediction = min_mc.cl

        # update micro clusters and save on disk
        if min_mc.cl == true_label:
            # scenario 1:
            #min_mc.cf_all.append(instance) #try not append all things


            min_mc.n = min_mc.n + 1
            #min_mc.centroid = np.add(min_mc.centroid * min_mc.n, np.array(features)) / min_mc.n

            temp_centroids = np.array(min_mc.centroid) * (min_mc.n - 1)
            min_mc.centroid = (np.add(temp_centroids, features_np) / min_mc.n).tolist()

            if min_mc.epsilon > 0:
                min_mc.epsilon -= 1
        else:
            # scenario 2:
            # need to find the true mc
            true_mc = self.find_true_nearest_mc(instance)

            #true_mc.cf_all.append(instance)

            true_mc.epsilon += 1
            true_mc.n = true_mc.n + 1
            #true_mc.centroid = np.add(true_mc.centroid * true_mc.n, np.array(features)) / true_mc.n
            temp_centroids = np.array(true_mc.centroid) * (true_mc.n - 1)
            true_mc.centroid = (np.add(temp_centroids, features_np) / true_mc.n).tolist()

            min_mc.epsilon += 1

            # TODO: check and split
            # ...

        # write updated mcs onto disk
        for mc in self.pool:
            if mc.cl == 'normal':
                with open(join(mc_folder, 'normal_mc_1.csv'), 'w', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([mc.epsilon]) #update new epsilon
                    csv_writer.writerow([mc.n])  # update new count
                    csv_writer.writerow(mc.centroid)  # update new centroids
                    #csv_writer.writerows(mc.cf_all)
            elif mc.cl == 'anomaly':
                with open(join(mc_folder, 'anomaly_mc_1.csv'), 'w', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([mc.epsilon])  # update new epsilon
                    csv_writer.writerow([mc.n])  # update new count
                    csv_writer.writerow(mc.centroid)  # update new centroids
                    #csv_writer.writerows(mc.cf_all)
            else:
                print("???????? non-reachable !!!!!!!!!!")

        return prediction


def init_mcnn_pool(data_file, sc):
    '''
    initialize the pool with 1 normal instance and 1 anomaly instance,
    write 2 instances to 2 files

    :param data_file: dataset
    :param folder: save folder
    :param sc:
    :return:
    '''
    # read the data
    data = sc.textFile(data_file).map(lambda x: list(reader(StringIO(x)))[0])

    normal = None
    anomaly = None
    while normal is None or anomaly is None:
        rand = data.takeSample(withReplacement=False, num=1, seed=random.randint(0, 100))[0]

        if rand[-1] == 'normal' and normal is None:
            normal = rand
        elif rand[-1] == 'anomaly' and anomaly is None:
            anomaly = rand

    with open(join(mc_folder, 'normal_mc_1.csv'), 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow("0")  # initial epsilon #first row
        csv_writer.writerow("1")  # initial count of instances in the cluster #second row
        csv_writer.writerow(normal)  # initial centroid in the cluster #third row
        #csv_writer.writerow(normal) #try not write everything in the cluster

    with open(join(mc_folder, 'anomaly_mc_1.csv'), 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow("0")  # initial epsilon #first row
        csv_writer.writerow("1")  # initial count of instances in the cluster #second row
        csv_writer.writerow(anomaly)  # initial centroid in the cluster #third row
        #csv_writer.writerow(anomaly)


def predict(instance):
    mcnn = MC_NN(theta=20)

    prediction = mcnn.predict_and_update_mcs(instance, instance[-1])

    # TODO: after the previous RDD deletes the files, the next RDD may fail to
    #       read the mc files.
    #clean_mc_folder()

    return None

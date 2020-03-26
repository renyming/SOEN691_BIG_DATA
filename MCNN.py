import os
import random
import csv
import shutil
import numpy as np
from io import StringIO
from os.path import join
from csv import reader
from mcnnmc import MC

from datetime import datetime

'''
改动的地方:
1. 每个mc文件头的信息：
    1) epsilon  : error counter
    2) n        : number of instances
    3) centroid : centroid
    4) CF2_x    : sum of the squares of the attributes in CF1_x
2. 每个rdd需要读：epsilon, n, centroid, CF2_x
    1) 先根据 centroid, n 算出 CF1_x
    2) 在根据 CF1_x, CF2_x 算出 vairiance
    3) 根据 centroid 算距离
    4) 不需要存每个点的数据
    5) Split : 找到最大的 variance 中的下标，先更新 CF1_x, 再根据 CF1_x 算出两个新的 
               centroid 写回去
    5) 如果用 foreachRDD，需要存下来:
          epsilon
          n 
          centroid (CF1_x / n)
          CF2_x
    6) 需要在 MCNN/predict 方法里把 (prediction, label, time) 写到一个新的文件
    7) 最后等所有 (prediction, label, time) 都存下来后，再写另一个py文件做 evaluation
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
        self.read_mcnn_pool() # populate the essential attributes for MC

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
                mc.n = int(lines[1])  # count
                mc.centroid = [float(x) for x in lines[2].split(',') if is_number(x)]
                mc.cf2_x = [float(x) for x in lines[3].split(',')]
                mc.status = lines[4]  # status
                mc.filename = lines[5]

            mc.cf1_x = [x * mc.n for x in mc.centroid]  # The sum of the feature is the average centroid * n

            if mc.status == 'activate':
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

    def split_mc(self):

        for mc in self.pool:

            if (mc.epsilon > self.theta):

                mc.status = 'deactivate'

                max_index = mc.variance_x.index(max(mc.variance_x))
                max_variance = mc.variance_x[max_index]

                child_mc_1 = MC(self.theta)
                child_mc_2 = MC(self.theta)

                child_mc_1.cl = mc.cl
                child_mc_2.cl = mc.cl

                child_mc_1.cf1_x = mc.cf1_x
                child_mc_2.cf2_x = mc.cf1_x

                child_mc_1.cf1_x[max_index] = max_variance
                child_mc_2.cf2_x[max_index] = -max_variance

                child_mc_1.centroid = [x / mc.n for x in child_mc_1.cf1_x]
                child_mc_2.centroid = [x / mc.n for x in child_mc_2.cf2_x]

                child_mc_1.cf2_x = [float(x) * float(y) for x, y in zip(child_mc_1.cf1_x, child_mc_1.cf1_x)]
                child_mc_2.cf2_x = [float(x) * float(y) for x, y in zip(child_mc_2.cf2_x, child_mc_2.cf2_x)]

                child_mc_1.status = 'activate'
                child_mc_2.status = 'activate'

                child_mc_1.filename = 'child_1_' + mc.filename
                child_mc_2.filename = 'child_2_' + mc.filename

                self.pool.append(child_mc_1)
                self.pool.append(child_mc_2)

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


            # update n
            min_mc.n = min_mc.n + 1

            # update centroids and cf1_x
            temp_cf1_x = np.array(min_mc.cf1_x)
            min_mc.cf1_x = np.add(temp_cf1_x, features_np)
            min_mc.centroid = (min_mc.cf1_x / min_mc.n).tolist()

            # update CF2_X
            temp_features = features_np * features_np
            min_mc.cf2_x = (np.add(min_mc.cf2_x, temp_features)).tolist()

            # calculate variance
            temp_cf2_x = np.array(min_mc.cf2_x) / min_mc.n
            temp_cf1_x_2 = np.multiply(min_mc.centroid, min_mc.centroid)
            min_mc.variance_x = np.sqrt(temp_cf2_x - temp_cf1_x_2).tolist()

            if min_mc.epsilon > 0:
                min_mc.epsilon -= 1
        else:
            # scenario 2:
            
            # need to find the true mc
            true_mc = self.find_true_nearest_mc(instance)

            # increment e
            true_mc.epsilon += 1
            min_mc.epsilon += 1

            # update n
            true_mc.n = true_mc.n + 1

            # update centroids and cf1_x
            temp_cf1_x = np.array(true_mc.cf1_x)
            true_mc.cf1_x = np.add(temp_cf1_x, features_np)
            true_mc.centroid = (true_mc.cf1_x / true_mc.n).tolist()

            # update CF2_X
            temp_features = np.multiply(features_np, features_np)
            true_mc.cf2_x = (np.add(true_mc.cf2_x, temp_features)).tolist()

            # calculate variance
            temp_cf2_x = np.array(true_mc.cf2_x) / true_mc.n
            temp_cf1_x_2 = np.multiply(true_mc.centroid, true_mc.centroid)
            true_mc.variance_x = np.sqrt(temp_cf2_x - temp_cf1_x_2).tolist()

            # TODO: check and split
            self.split_mc()

        # write updated mcs onto disk
        for mc in self.pool:

            with open(join(mc_folder, mc.filename), 'w', newline='') as f:

                csv_writer = csv.writer(f)
                csv_writer.writerow([mc.epsilon])  # write new epsilon
                csv_writer.writerow([mc.n])  # write new count
                csv_writer.writerow(mc.centroid)  # write new centroids
                csv_writer.writerow(mc.cf2_x)  # write new CF2_X
                csv_writer.writerow([mc.status])
                csv_writer.writerow([mc.filename])

        # return type (prediction_lable, true_lable), time_stamp)
        # need more formal format for the time_stamp
        return prediction, instance[-1], datetime.now()

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
    cf2_x_normal = None
    cf2_x_anomaly = None

    while normal is None or anomaly is None:
        rand = data.takeSample(withReplacement=False, num=1, seed=random.randint(0, 100))[0]

        if rand[-1] == 'normal' and normal is None:
            normal = rand

            cf1_x_normal = [x for x in normal if is_number(x)]
            cf2_x_normal = [float(x) * float(y) for x, y in zip(cf1_x_normal, cf1_x_normal)]

        elif rand[-1] == 'anomaly' and anomaly is None:
            anomaly = rand
            cf1_x_anomaly = [x for x in anomaly if is_number(x)]
            cf2_x_anomaly = [float(x) * float(y) for x, y in zip(cf1_x_anomaly, cf1_x_anomaly)]

    with open(join(mc_folder, 'normal_mc_1.csv'), 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow("0")           # initial epsilon #first row
        csv_writer.writerow("1")           # initial count of instances in the cluster #second row
        csv_writer.writerow(normal)        # initial centroid in the cluster #third row
        csv_writer.writerow(cf2_x_normal)  # initial cf2_x
        csv_writer.writerow(['activate'])  # initial mc status
        csv_writer.writerow(['normal_mc_1.csv'])

    with open(join(mc_folder, 'anomaly_mc_1.csv'), 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow("0")            # initial epsilon #first row
        csv_writer.writerow("1")            # initial count of instances in the cluster #second row
        csv_writer.writerow(anomaly)        # initial centroid in the cluster #third row
        csv_writer.writerow(cf2_x_anomaly)  # initial cf2_x
        csv_writer.writerow(['activate'])       # initial mc status
        csv_writer.writerow(['anomaly_mc_1.csv'])


def predict(instance):

    mcnn = MC_NN(theta=20)

    (prediction, true_lable, time_stamp) = mcnn.predict_and_update_mcs(instance, instance[-1])

    # TODO: after the previous RDD deletes the files, the next RDD may fail to
    #       read the mc files.
    #Save the pair of (prediction, true_lable) to the disk for the future evaluation

    #clean_mc_folder()


    return None

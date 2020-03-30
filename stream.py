import csv
import time
import math
import os, shutil


def start(nb_instances, sleep_time):
    raw_data = []
    with open('./source_dir/Train_c.csv', 'r') as csv_file:

        csv_reader = csv.reader(csv_file)

        for line in csv_reader:
            raw_data.append(line)

    total_size_data = len(raw_data)
    nb_instance_per_batch = nb_instances
    nb_batches = math.ceil(total_size_data / nb_instance_per_batch)

    start_index = 0

    for i in range(nb_batches):

        with open('./input_dir/raw_data' + str(i) + '.csv', 'w', newline='') as file:

            csv_writer = csv.writer(file)

            if total_size_data >= nb_instance_per_batch:
                nb_instance_per_batch = nb_instances
            else:
                nb_instance_per_batch = total_size_data % nb_instances

            for j in range(nb_instance_per_batch):

                csv_writer.writerow(raw_data[j + start_index])

        start_index += nb_instance_per_batch
        total_size_data -= nb_instance_per_batch
        time.sleep(sleep_time)

def clean():

    folder = ("./input_dir")
    for filename in os.listdir(folder):

        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def start_stream():
    clean()
    start(100, 3)

start_stream()
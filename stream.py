import csv
import time
import os, shutil, glob
import matplotlib.pyplot as pyplot


def start():

    raw_data = []
    with open('./source_dir/Train.csv', 'r') as csv_file:

        csv_reader = csv.reader(csv_file)

        for line in csv_reader:
            raw_data.append(line)


    start_index = 0

    for i in range(200):

        with open('./input_dir/raw_data' + str(i) + '.csv', 'w', newline='') as file:

            csv_writer = csv.writer(file)
            #ri = random.randint(40,50)

            ri = 100

            for j in range(ri):

                csv_writer.writerow(raw_data[j + start_index])

        start_index += ri

        time.sleep(3)

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


def cleanRTKNN():

    folder = ("./rt_knn_dir")
    for filename in os.listdir(folder):

        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def calculateResults():

    print('--- Calculating the F1 Score ---------')
    f1_score_list = []
    folder = ("./rt_knn_dir")
    for filename in glob.glob(os.path.join(folder, '*.txt')):

        with open(filename, 'r') as f:
            results = []

            true_positive = 0
            true_negitive = 0
            false_positive = 0
            false_negitive = 0

            csvreader = csv.reader(f)
            for row in csvreader:
                results.append(row)

            for result in results:
                true_label = result[0]
                predicted_label = result[1]

                if true_label == 'normal' and predicted_label == 'normal':
                    true_positive = true_positive + 1

                elif true_label == 'anomaly' and predicted_label == 'anomaly':
                    true_negitive = true_negitive + 1

                elif true_label == 'normal' and predicted_label == 'anomaly':
                    false_positive = false_positive + 1
                else:
                    false_negitive = false_negitive + 1

            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negitive)

            # F1 Score
            f1_score = 2 * ((precision * recall) / (precision + recall))
            f1_score_list.append(f1_score)

    # print(len(f1_score_list))

    x = [i for i in range(10, (len(f1_score_list) + 1) * 10, 10)]

    fig, ax = pyplot.subplots()

    ax.plot(x, f1_score_list)
    pyplot.ylabel('F1 Score')
    pyplot.title('RT KNN (50)')
    pyplot.show()





clean()

cleanRTKNN()

start()

calculateResults()



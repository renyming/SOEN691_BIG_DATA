import csv
import time
import os, shutil

def start():

    raw_data = []
    with open('./source_dir/Train_c.csv', 'r') as csv_file:

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

clean()

start()





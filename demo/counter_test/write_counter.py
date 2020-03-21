import time

counter = 1

while True:
    with open('./counter/count_' + str(counter) + '.txt', 'w') as f:
        f.write(str(counter))
        f.close()

    counter += 1

    time.sleep(3)


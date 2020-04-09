import numpy as np
import pandas as pd
# importing the required module
import matplotlib.pyplot as plt
import csv

def mcnn_eval():
    # read results in 'mcnn_predictions.csv'
    with open('./mcnn_pred/mcnn_predictions.csv', 'r') as f:
        lines = f.read().splitlines()
        lines = [x.split(',') for x in lines]
        lines_np = np.array(lines)

        pred = lines_np[:, 0]
        true = lines_np[:, 1]
        time = lines_np[:, 2].astype(np.int)
        time -= time[0]

        # anomaly: positive
        # normal: negative
        true_pos = np.sum(np.logical_and(pred == 'anomaly', true == 'anomaly'))
        false_pos = np.sum(np.logical_and(pred == 'anomaly', true == 'normal'))
        true_neg = np.sum(np.logical_and(pred == 'normal', true == 'normal'))
        false_neg = np.sum(np.logical_and(pred == 'normal', true == 'anomaly'))

        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)

        print("Accuracy: {:0.5f}".format(accuracy))
        print("Precision: {:0.5f}".format(precision))
        print("Recall: {:0.5f}".format(recall))
        print("F1 Measure: {:0.5f}".format((2 * precision * recall / (precision + recall))))


def mcnn_prequential_error_plot():

    headers = ['error_count', 'n_count', 'prequential_error']
    df1 = pd.read_csv('./mcnn_pred/2-3000.csv', names=headers)
    df2 = pd.read_csv('./mcnn_pred/10-3000.csv', names=headers)


    x1 = df1['n_count']
    x2 = df2['n_count']
    y1 = df1['prequential_error']
    y2 = df2['prequential_error']

    plt.plot(x1, y1, 'r', label = 'theta = 2')
    plt.plot(x2, y2, 'g', label = 'theta = 10')

    # naming the x axis
    plt.xlabel('data counts')
    # naming the y axis
    plt.ylabel('prequential_error')
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.show()

mcnn_prequential_error_plot()


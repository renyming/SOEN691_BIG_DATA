import numpy as np

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


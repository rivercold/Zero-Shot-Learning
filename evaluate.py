import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def get_metrics(predictions, labels):
    # predictions: matrix of (test_size * 200), each cell is the probability of classified as 1
    # labels: vector of (test_size * 1), each cell is the true label of test image
    test_size = labels.shape[0]
    class_num = predictions.shape[1]

    true_labels = np.zeros(shape=predictions.shape)
    true_labels[xrange(test_size), labels] = 1

    ## roc_auc and ap
    roc_auc = np.zeros(shape=(1, class_num))
    ap = np.zeros(shape=(1, class_num))

    for i in xrange(class_num):
        roc_auc[i] = roc_auc_score(true_labels[:, i], predictions[:, i]) # default macro
        ap[i] = average_precision_score(true_labels[:, i], predictions[:, i]) # default macro


    ## top 1 and top 5 accuracy
    top1 = np.argmax(predictions, axis=1)
    top1_accu = np.sum(top1 == labels) / float(test_size)

    top5 = np.argsort(predictions, axis=1)[:, -5:]
    top5_accu = np.sum(top5 == label) / float(test_size)

    return (roc_auc, ap, top1_accu, top5_accu)
    

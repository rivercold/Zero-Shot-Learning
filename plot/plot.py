import numpy as np
import matplotlib.pyplot as plt

log_file = '../log/fc_BCE_tmlp_6366-300-50_vmlp_1000-200-50_bs_200_1109-00-52-51.log'

seen = [[],[],[],[]]
mean = [[],[],[],[]]
unseen = [[],[],[],[]]

with open(log_file, 'r') as f:
    while True:
        line1 = f.readline().rstrip()
        line2 = f.readline().rstrip()
        line3 = f.readline().rstrip()
        line4 = f.readline().rstrip()
        line5 = f.readline().rstrip()

        if not line1: break  # EOF

        for i in xrange(4):
            seen[i].append(float(line3.split('\t')[i + 2].split(' ')[-1]))
            unseen[i].append(float(line4.split('\t')[i + 2].split(' ')[-1]))
            mean[i].append(float(line5.split('\t')[i + 2].split(' ')[-1]))

x = xrange(1, 1 + len(seen[0]))

metrics = ['ROC-AUC', 'PR-AUC', 'Top-1 Acc', 'Top-5 Acc']

for i, metric in enumerate(metrics):
    fig = plt.figure(i, figsize=(10, 5))
    fig.set_size_inches(10, 5)

    plt.plot(x, seen[i], 'r', label='%s, seen' % metric)
    plt.plot(x, unseen[i], 'g', label='%s, unseen' % metric)
    plt.plot(x, mean[i], 'b', label='%s, mean' % metric)

    plt.title('%s wrt epoch' % metric)# give plot a title
    plt.xlabel('Epoch')# make axis labels
    plt.ylabel(metric)

    plt.xlim(x[0], x[-1])# set axis limits
    plt.ylim(0.0, 1.0)

    plt.legend(loc=0)

    plt.savefig('figures/%s.png' % metric)

    plt.show()# show the plot on the screen
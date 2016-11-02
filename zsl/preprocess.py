__author__ = 'yuhongliang324'

from load_data import *
import random


# return: split result:
# 1 - training, 0 - seen test, -1 - unseen test
def split(matroot, split_file, unseen_file):
    Y = None
    files = os.listdir(matroot)
    files.sort()
    for fn in files:
        if not fn.endswith('.mat'):
            continue
        data = loadmat(os.path.join(matroot, fn))
        features = data['fc7']
        label = int(fn[:3]) - 1
        y = label * numpy.ones((features.shape[0],))
        if Y is None:
            Y = y
        else:
            Y = numpy.concatenate((Y, y))

    sp = numpy.ones_like(Y)

    remain = []

    test_classes = random.sample(range(num_class), 40)
    test_classes.sort()
    for i in xrange(Y.shape[0]):
        if Y[i] in test_classes:
            sp[i] = -1
        else:
            remain.append(i)
    test_index = random.sample(remain, int(len(remain) * 0.2))
    for ti in test_index:
        sp[ti] = 0

    writer = open(split_file, 'w')
    for i in xrange(sp.shape[0]):
        writer.write(str(int(sp[i])) + '\n')
    writer.close()

    writer = open(unseen_file, 'w')
    for tc in test_classes:
        writer.write(str(tc) + '\n')
    writer.close()


if __name__ == '__main__':
    split('../features/bird-2010/resnet', '../features/bird-2010/zsl_split.txt',
          '../features/bird-2010/unseen_classes.txt')

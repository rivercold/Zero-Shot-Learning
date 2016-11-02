__author__ = 'yuhongliang324'

from load_data import *
import random


def split(matroot):
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
    test_classes = numpy.asarray(random.sample(range(num_class), 40))


split('../features/resnet')

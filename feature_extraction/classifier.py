__author__ = 'yuhongliang324'

from scipy.io import loadmat, savemat
import os
import numpy


def vgg_preprocess(vgg_root, feature_root):

    files = os.listdir(vgg_root)
    files.sort()
    for fn in files:
        dirpath = os.path.join(vgg_root, fn)
        if not os.path.isdir(dirpath):
            continue
        features = None
        mats = os.listdir(dirpath)
        for mat in mats:
            matpath = os.path.join(dirpath, mat)
            if not mat.endswith('.mat'):
                continue
            data = loadmat(matpath)
            data = data['feats']
            fc7 = data[0][0][1]
            fc7 = numpy.mean(fc7, axis=1)
            fc7 = numpy.reshape(fc7, (1, 4096))
            if features is None:
                features = fc7
            else:
                features = numpy.concatenate((features, fc7), axis=0)

        mdict = {'fc7': features}

        savemat(os.path.join(feature_root, fn) + '.mat', mdict)


if __name__ == '__main__':
    vgg_root = '/usr0/home/hongliay/zsl/data/CUB_200_2011/vgg'
    feature_root = '/usr0/home/hongliay/code/Zero-Shot-Learning/features'
    vgg_preprocess(vgg_root, feature_root)

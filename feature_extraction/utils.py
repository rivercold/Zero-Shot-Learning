__author__ = 'yuhongliang324'
import numpy

word_vector_file = '../../word_vectors/glove.840B.300d.txt'


def form_vectors(line):
    sp = line.split(' ')
    word = line[0]
    sp = sp[1:]
    vec = [float(x) for x in sp]
    vec = numpy.array(vec, dtype=numpy.float)
    return word, vec


def get_all_vectors(wv_file):
    word_vec = {}
    reader = open(wv_file)
    count = 0
    while True:
        line = reader.readline()
        if line:
            count += 1
            if count % 10000 == 0:
                print count
            line = line.strip()
            word, vec = form_vectors(line)
            word_vec[word] = vec
        else:
            break
    reader.close()

get_all_vectors(word_vector_file)

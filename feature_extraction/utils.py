__author__ = 'yuhongliang324'
import numpy
import pickle
import os

word_vector_file = '../../../word_vectors/glove.840B.300d.txt'


def form_vectors(line):
    sp = line.split(' ')
    word = sp[0]
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
    return word_vec


def write_voc(wv_file, voc_file, out_pkl, oov_file):

    assert os.path.isfile(wv_file)
    assert os.path.isfile(voc_file)

    word_vec_all = get_all_vectors(wv_file)

    reader = open(voc_file)
    words = reader.readlines()
    print len(words)
    reader.close()

    words = map(lambda x: x.strip(), words)

    writer_oov = open(oov_file, 'w')
    word_vec = {}
    for word in words:
        if not word in word_vec_all:
            writer_oov.write(word + '\n')
            continue
        word_vec[word] = word_vec_all[word]
    writer_oov.close()

    with open(out_pkl, 'wb') as fin:
        pickle.dump(word_vec, fin)
    print len(word_vec)


def test1():
    write_voc(word_vector_file, '../wiki/api_extracted/first50_vocab.txt',
              '../features/summary/vocab.pkl', '../features/summary/oov.txt')


if __name__ == '__main__':
    test1()

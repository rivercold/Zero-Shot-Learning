from lxml import etree
from collections import defaultdict
import urllib2
import time, random,sys
import re, os, string, pickle
from readability import Document
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

reload(sys)
sys.setdefaultencoding('utf8')
oov_list = []
# match a token into words in vocabulary

def transform_to_words(token,vocab,long_vocab):
    if token in vocab:
        return [token]
    else:
        token_list = []
        flag = 0
        for id,char in enumerate(token):
            if token[flag:id+1] in long_vocab:
                token_list.append(token[flag:id+1])
                flag = id+1

        if flag == len(token):
            print token, token_list
            return token_list
        else:
            return ["oov"]

# only use word that has at least 3 chars
def transform_vocab(vocab):
    long_vocab = []
    for word in vocab.keys():
        if len(word) >= 3:
            long_vocab.append(word)
    return long_vocab


def generate_average(vocab):
    mat = [vec for key,vec in vocab.iteritems()]
    mat = np.array(mat)
    print mat.shape
    return np.mean(mat,axis=0)

# parse and match the summary file with the vocabulary
# return vector matrix / tensor
if __name__ == "__main__":

    file_path = open("new_summary.txt","r")
    write_file = open("../../features/summary/summary_feat","w")
    vocab_file = open("../../features/summary/vocab.pkl","rb")
    vocab = pickle.load(vocab_file)
    average = generate_average(vocab)

    lines = file_path.readlines()
    tensor = []
    for index, line in enumerate(lines):
        mat = []
        tokens = line.split()
        if len(tokens) < 30:
            tokens += [" oov" for i in range(len(tokens),30)]
            print tokens
        print index, len(tokens)
        for id in range(30):
            word = tokens[id]
            if word in vocab:
                vec = vocab[word]
            else:
                vec = average
            mat.append(vec)
        tensor.append(mat)

    tensor = np.array(tensor)
    print type(tensor)
    np.save(write_file,tensor)
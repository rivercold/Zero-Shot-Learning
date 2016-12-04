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
        # 04 oz like that
        if chr(194) in token:
            print token.split(chr(194))
            return token.split(chr(194))
        else:
            flag = 0
            for id,char in enumerate(token):
                if token[flag:id+1] in long_vocab:
                    token_list.append(token[flag:id+1])
                    flag = id+1
            print token, token_list
            if token not in oov_list:
                oov_list.append(token)
            return token_list

# only use word that has at least 3 chars
def transform_vocab(vocab):
    long_vocab = []
    for word in vocab.keys():
        if len(word) >= 3:
            long_vocab.append(word)
    return long_vocab


# parse and match the summary file with the vocabulary
# return vector matrix / tensor
if __name__ == "__main__":

    file_path = open("summary.txt","r")
    write_file = open("new_summary.txt","w")
    vocab_file = open("../../features/summary/vocab.pkl","rb")
    vocab = pickle.load(vocab_file)
    long_vocab = transform_vocab(vocab)
    #print vocab["summer"]
    #print vocab["dark"]
    #raise

    lines = file_path.readlines()
    for index, line in enumerate(lines):
        tokens = line.split()
        words = []
        for id, token in enumerate(tokens):
            words += transform_to_words(token,vocab,long_vocab)
            if len(words) >= 30:
                break
        words = words[:30]
        write_file.write(" ".join(words)+"\n")
    oov_set = set(oov_list)
    print len(oov_set)
    print "========="
    for token in oov_set:
        print token

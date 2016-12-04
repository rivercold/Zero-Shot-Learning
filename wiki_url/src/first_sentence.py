from lxml import etree
from collections import defaultdict
import urllib2
import time, random,sys
import re, os, string, pickle
from readability import Document
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

reload(sys)
sys.setdefaultencoding('utf8')

if __name__ == "__main__":
    path = "raw_wiki.txt"
    results_file = "first_sentence.txt"
    write_file = open("summary.txt","w")

    lines = open(results_file,"r").readlines()
    for index, line in enumerate(lines):
        words = line.split()
        for id, word in enumerate(words):
            if word == "wikipedia":
                wid = id
                break
        words =  words[wid+1:wid+30]
        print words
        write_file.write(" ".join(words)+"\n")


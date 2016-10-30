from lxml import etree
from collections import defaultdict
import urllib2
import time, random,sys
import re, os, string
from readability import Document
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

reload(sys)
sys.setdefaultencoding('utf8')

class parser:
    def __init__(self):
        self.name = defaultdict(list)
        self.url_list = self.get_url_list()
        self.wiki_path = "../wiki_url"


    def get_url_list(self):
        path = "../wiki_url/url.txt"
        url_list = []
        for line in open(path,'r').readlines():
            url_list.append(line.strip())
        #print url_list
        return url_list


    def crawl(self):
        for index, url in enumerate(self.url_list):
            print url
            response = urllib2.urlopen(url, timeout=30)
            lines = response.read()
            file_name = "{}/html/{}".format(self.wiki_path,url.replace("/","_"))
            write_file = open(file_name, 'w')
            write_file.write(lines)
            random_time_s = random.randint(5, 10)
            time.sleep(random_time_s)
            if index%10 == 9:
                random_time_s = random.randint(40, 90)
                time.sleep(random_time_s)

    @staticmethod
    def strip (script):
        rc_tag = re.compile(r'(?<=<).*?(?=>)')
        tags = rc_tag.findall(script)
        for tag in tags:
            script = script.replace('<'+tag+'>', '')
        script = script.replace('\n','')
        script = script.replace('\t','')
        return script

    def parse(self,file_path):
        with open(file_path) as fin:
            content = fin.read()
        doc = Document(content)
        title = doc.title()
        article = doc.summary()
        readable_article=self.strip(article)
        readable_title=self.strip(title)
        #print readable_article
        #print readable_title
        return readable_title + " " + readable_article

    def get_corpus(self):
        folder = "../wiki_url/html/"
        corpus = []
        index = 0
        for file in os.listdir(folder):
            file_path = folder + file
            text = self.parse(file_path)
            text = self.preprocess(text)
            corpus.append(text)
            index += 1
            if index > 200:
                break
        print corpus[0]
        print len(corpus)
        raw_text_file = open("raw_wiki.txt","w")
        for text in corpus:
            text = text.encode('utf8')
            raw_text_file.write(text+"\n")
        self.corpus = corpus

    def extract_features(self,corpus):
        vectorizer = TfidfVectorizer(min_df=3,token_pattern='\\b\\w+\\b')
        features = vectorizer.fit_transform(corpus)
        self.vectorizer = vectorizer
        return features


    def preprocess(self,text):
        t = text.lower()
        exclude = set(string.punctuation)
        t = t.replace("["," ")
        t = t.replace("]"," ")
        t = t.replace("-"," ")
        s = ''.join(ch for ch in t if ch not in exclude)
        return s


if __name__ == "__main__":
    #path = "../wiki_url/html/http:__en.wikipedia.org_wiki_Hooded_warbler"
    p = parser()
    p.get_corpus()
    feat = p.extract_features(p.corpus)
    print type(feat)
    print feat.shape
    np.save("./wiki_features",feat)

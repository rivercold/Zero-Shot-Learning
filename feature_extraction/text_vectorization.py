__author__ = 'yuhongliang324'
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy


def load_stopwords():
    stopword_file = '../stopword.list'
    reader = open(stopword_file)
    words = reader.readlines()
    reader.close()
    words = map(lambda x: x.strip(), words)
    return set(words)


def contains_digit(word):
    for ch in word:
        if ch in '0123456789':
            return True
    return False

sw = load_stopwords()


def preprocess_text(line):
    line = line.lower()
    line = line.decode('utf-8')
    line = ' '.join(word_tokenize(line))
    line = ' '.join([word for word in line.split() if word not in sw])  # remove stopwords
    line = ' '.join([word for word in line.split() if not contains_digit(word)])  # remove digits
    return line


def get_tfidf(raw_file, wiki_npy, voc_file, min_df=3):
    reader = open(raw_file)
    lines = reader.readlines()
    reader.close()
    lines = map(lambda x: x.strip(), lines)
    corpus = map(preprocess_text, lines)

    vectorizer = CountVectorizer(min_df=min_df)
    X = vectorizer.fit_transform(corpus)
    X = X.toarray()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    tfidf = tfidf.toarray()
    voc = vectorizer.get_feature_names()
    print len(voc)

    numpy.save(wiki_npy, tfidf)
    writer = open(voc_file, 'w')
    for word in voc:
        writer.write(word.encode('utf-8') + '\n')
    writer.close()


def test1():
    get_tfidf('../wiki/api_extracted/full_filtered.txt',
              '../features/wiki/wiki.npy', '../features/wiki/vocabulary.txt')

if __name__ == '__main__':
    test1()

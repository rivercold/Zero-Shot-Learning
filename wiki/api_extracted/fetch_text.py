import urllib, json
import re
import nltk
import pickle
import numpy as np

def fetch_text():
    with open('full.txt', 'w') as f_full:
        with open('url.txt', 'r') as f:
            for line in f:
                class_name = line.rstrip().split('/')[-1]

                # intro_url = 'https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exlimit=max&explaintext&exintro&titles=%s&redirects=' % class_name
                # intro_response = urllib.urlopen(intro_url)
                # intro_data = json.loads(intro_response.read())

                full_url = 'https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exlimit=max&explaintext&titles=%s&redirects=' % class_name
                full_response = urllib.urlopen(full_url)
                full_data = json.loads(full_response)
                
                key = full_data["query"]["pages"].keys()[0]
                extract = full_data["query"]["pages"][key]["extract"]

                f_full.write(extract.replace('\n', ' ').encode('utf-8') + '\n')


def extract_section():
    regex = ur' ={2,4} ([\w ]+) ={2,4} '
    all_sections = dict()

    with open('full.txt', 'r') as f:
        for line in f:
            sections = re.findall(regex, line)
            
            for s in sections:
                if s not in all_sections:
                    all_sections[s] = 1
                else:
                    all_sections[s] += 1
            # all_sections = all_sections.union(sections)

    # print all_sections
    with open('all_sections.txt', 'w') as f_full:
        for w in sorted(all_sections, key=all_sections.get, reverse=True):
            # print w, d[w]
            f_full.write('%d %s' % (all_sections[w], w) + '\n')


def filter_text():
    regex = r' ={2,4} [\w ]+ ={2,4} '
    regex_title = r' ={2,4} ([\w ]+) ={2,4} '

    filtered_titles = set()
    with open('filter_titles.txt', 'r') as f:
        for line in f:
            filtered_titles.add(line.rstrip())

    all_sections = dict()

    with open('full_filtered.txt', 'w') as f_writer:
        with open('full.txt', 'r') as f:
            for line in f:
                cur_sections = re.findall(regex_title, line)
                cur_contents = re.split(regex, line.rstrip())

                for idx, s in enumerate(cur_sections):
                    if s in filtered_titles:
                        cur_contents[idx + 1] = ''

                new_contents = ' '.join(cur_contents)
                f_writer.write(new_contents + '\n')

    # print all_sections
    with open('all_sections.txt', 'w') as f_full:
        for w in sorted(all_sections, key=all_sections.get, reverse=True):
            # print w, d[w]
            f_full.write('%d %s' % (all_sections[w], w) + '\n')


def tokenize():
    first50_vocab = set()

    with open('full_filtered.txt', 'r') as f:
        for line in f:
            tokens = nltk.word_tokenize(line.rstrip().lower().decode('utf-8'))[:50]
            first50_vocab = first50_vocab.union(tokens)

    with open('first50_vocab.txt', 'w') as f:
        for v in first50_vocab:
            f.write(v.encode('utf-8') + '\n')

def find_oov_with_dash():
    oov_file = '../../features/summary/oov.txt'

    oov_with_dash = []

    with open(oov_file, 'r') as f:
        for line in f:
            cur = line.rstrip().decode('utf-8')
            if '-' in cur or u'\u2013' in cur:
                oov_with_dash.append(cur)

    with open('oov_with_dash.txt', 'w') as f:
        for oov in oov_with_dash:
            f.write(oov.encode('utf-8') + '\n')


def retokenize():
    oov_with_dash = set()
    with open('oov_with_dash.txt', 'r') as f:
        for line in f:
            oov_with_dash.add(line.rstrip().decode('utf-8').replace(u'\u2013', '-'))

    first50_vocab_without_oov_with_dash = set()
    with open('first50_vocab.txt', 'r') as f:
        for line in f:
            cur = line.rstrip().decode('utf-8').replace(u'\u2013', '-')
            if cur in oov_with_dash:
                contents = cur.split('-')
                for c in contents:
                    first50_vocab_without_oov_with_dash.add(c)
            else:
                first50_vocab_without_oov_with_dash.add(cur)

    # removed oov with dash in our old first50_vocab.txt file
    with open('first50_vocab_new.txt', 'w') as f:
        for v in first50_vocab_without_oov_with_dash:
            f.write(v.encode('utf-8') + '\n')


def word2vec():
    pickle_file = '../../features/summary/vocab.pkl'
    vec_map = pickle.load(open(pickle_file))

    # Find the average vector
    avg_vec = np.zeros((300,))
    word_num = len(vec_map)
    for _, vec in vec_map.items():
        avg_vec += vec

    avg_vec /= word_num


    # xxx
    tensor = np.zeros((200, 50, 300))
    with open('full_filtered.txt', 'r') as f:
        for idx, line in enumerate(f):
            vectors = np.zeros((100,300))

            tokens = nltk.word_tokenize(line.rstrip().lower().decode('utf-8'))[:50]

            counter = 0

            for token in tokens:
                if token in vec_map:
                    vectors[counter] = vec_map[token]
                    counter += 1
                elif '-' in token or u'\u2013' in token:
                    if '-' in token:
                        temp = token.split('-')
                    else:
                        temp = token.split(u'\u2013')

                    for t in temp:
                        if t in vec_map:
                            vectors[counter] = vec_map[t]
                            counter += 1
                        else:
                            vectors[counter] = avg_vec
                            counter += 1
                else:
                    vectors[counter] = avg_vec
                    counter += 1

                    # print token

            vectors = vectors[:50]

            tensor[idx] = vectors

    # np.save('../../features/summary/tensor', tensor)


if __name__ == '__main__':
    # fetch_text()

    # extract_section()

    # filter_text()

    # tokenize()

    # find_oov_with_dash()

    # retokenize()

    word2vec()
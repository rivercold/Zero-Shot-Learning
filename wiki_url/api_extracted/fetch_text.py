import urllib, json
import re

def extract_text():
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
    with open('section.txt', 'w') as f_full:
        for w in sorted(all_sections, key=all_sections.get, reverse=True):
            # print w, d[w]
            f_full.write('%d %s' % (all_sections[w], w) + '\n')



if __name__ == '__main__':
    # extract_text()

    extract_section()
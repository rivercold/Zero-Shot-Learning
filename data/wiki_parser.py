from lxml import etree
from collections import defaultdict
import urllib2
import time, random


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

    def parse(self,file_path):
        with open(file_path) as fin:
            content = fin.read()
        #print file_path
        text = ""
        page_tree = etree.HTML(content)
        nodes = page_tree.xpath("//a/text()")
        text  = " ".join( [ node for node in nodes ] )


if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Hooded_warbler"
    p = parser()
    p.crawl()

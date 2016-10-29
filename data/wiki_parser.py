from lxml import etree
from collections import defaultdict
import urllib2


class parser:
    def __init__(self):
        self.name = defaultdict(list)


    def crawl(self,url):
        response = urllib2.urlopen(url, timeout=30)
        content = response.read()
        page_tree = etree.HTML(content)
        nodes = page_tree.xpath("//text()")
        text  = " ".join( [ node for node in nodes ] )
        print text
        return text


if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Hooded_warbler"
    p = parser()
    p.crawl(url)

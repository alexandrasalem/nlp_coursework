import nltk
from lxml import etree
import os
import sys

def main(filename):
    print("working on: " + filename)
    with open ("deserialized.txt", "a") as f:
        tree = etree.parse(filename)
        for doc in tree.xpath('DOC', type = "story"):
            story = doc.xpath('TEXT/P')
            for actual_story in story:
                if actual_story.text != None:
                    print(actual_story.text)
                    f.write(actual_story.text + " ")


for root, dirs, files in os.walk('./cna_eng'):
   for filename in files:
       if ".xml.gz" in filename:
           main('./cna_eng/' + str(filename))



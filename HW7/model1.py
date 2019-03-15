#!/usr/bin/env python
# model1.py: Model 1 translation table estimation
# Steven Bedrick <bedricks@ohsu.edu> and Kyle Gorman <gormanky@ohsu.edu>

from collections import defaultdict

def bitext(source, target):
    """
    Run through the bitext files, yielding one sentence at a time
    """
    for (s, t) in zip(source, target):
        yield ([None] + s.strip().split(), t.strip().split())
        # by convention, only target-side tokens may be aligned to a null
        # symbol (here represented with None) on the source side


class Model1(object):
    """
    IBM Model 1 translation table
    """

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    def __init__(self, source, target):
        self.source = source
        self.target = target
        #raise NotImplemented

    def train(self, n):
        """
        Perform n iterations of EM training
        """
        source_language_text = open(self.source, "r") 
        target_language_text = open(self.target, "r")
        
        sentences = bitext(source_language_text, target_language_text)
        
        #initializing dictionaries
        translation_table = defaultdict(lambda: defaultdict(float))
        for sentence in sentences:
            for x in sentence[0]:
                for y in sentence[1]:
                    translation_table[x][y] = 1
                    
        for key in translation_table:
            x = translation_table[key]
            length = len(translation_table.keys())
            for value in x:
                x[value] = 1/length
                

                
        print("done translation table")
        

        #main algorithm
        for i in range(n):
            print(i)
            #initializing counts
            source_language_text = open(self.source, "r") 
            target_language_text = open(self.target, "r")
            sentences = bitext(source_language_text, target_language_text)
            
            count_source_target = defaultdict(lambda: defaultdict(float))            
  
            total_target = defaultdict(float) 
            s_total = defaultdict(float)
            for sentence in sentences:
                #print(sentence)
                #normalization
                for x in sentence[0]:
                    for y in sentence[1]:
                        #print(x)
                        #print(y)
                        #print(translation_table[x][y])
                        s_total[x] += translation_table[x][y]
                #print(s_total)
                
                #counts
                for x in sentence[0]:
                    for y in sentence[1]:
                        count_source_target[x][y] += (translation_table[x][y]/s_total[x])
                        total_target[y] += (translation_table[x][y]/s_total[x])
                #print(count_source_target)
                #print(total_target)
                
            source_language_text = open(self.source, "r") 
            target_language_text = open(self.target, "r")
            sentences = bitext(source_language_text, target_language_text)
            
            for sentence in sentences:
                for x in sentence[0]:
                    for y in sentence[1]:
                        translation_table[x][y] = (count_source_target[x][y]/total_target[y])

        return translation_table
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()

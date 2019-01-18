from nltk import tokenize
from nltk import corpus
from collections import defaultdict
import string
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Creating unigrams, counting them

def unigrams(filename):
    print(filename + ':')
    unigrams = defaultdict(float)
    tokens = 0
    with open(filename, "r") as file:
        for line in file:
            split_line = line.split()
            for word in split_line:
                unigrams[word] += 1
                tokens+=1
        #print("working")
    #print(unigrams)
    print("number of types: " + str(len(unigrams)))
    print("number of tokens: " + str(tokens))

    sorted_unigrams = sorted(unigrams, key=unigrams.get, reverse=True)
    top_twenty_unigrams = sorted_unigrams[:20]
    print("top twenty unigrams: " + str(top_twenty_unigrams))
    return unigrams

#--------------------------------------------------------------------

# Creating bigrams

def bigrams(filename):
    print(filename + ':')
    bigrams = defaultdict(float)
    with open(filename, "r") as file:
        for line in file:
            split_line = line.split()
            for i in range(0, len(split_line)-1):
                word_one = split_line[i]
                word_two = split_line[i+1]
                bigram = (word_one, word_two)
                bigrams[bigram] +=1

    sorted_bigrams = sorted(bigrams, key=bigrams.get, reverse=True)
    top_twenty_bigrams = sorted_bigrams[:20]
    print("top twenty bigrams: " + str(top_twenty_bigrams))
    return bigrams

#--------------------------------------------------------------------

# PMI calculation

def all_pmi(unigrams, bigrams):
    tokens = sum(unigrams.values())
    all_pmi = defaultdict(float)
    for key in bigrams.keys():
        #print(key)
        bigram_prob = bigrams[(key[0], key[1])]/unigrams[key[0]]
        word_two_prob = unigrams[key[1]]/tokens
        all_pmi[key] = (bigram_prob/word_two_prob)
    return all_pmi


uni = unigrams("tokenized_with_stopwords.txt")
uni_without_stopwords = unigrams("tokenized_without_stopwords.txt")
bi = bigrams("tokenized_with_stopwords.txt")
all_pmi = all_pmi(uni, bi)

sorted_pmi = sorted(all_pmi, key=all_pmi.get, reverse=True)
top_thirty_pmi = sorted_pmi[:30]
print("top thirty pmi pairs: " + str(top_thirty_pmi))
print("unigram count for GREEN-COLORED: " + str(uni['GREEN-COLORED']))
print("unigram count for SHIPS: " + str(uni['SHIPS']))
print("bigram count for (GREED-COLORED SHIPS)" + str(bi[('GREEN-COLORED', 'SHIPS')]))

print("unigram count for NEW: " + str(uni['NEW']))
print("unigram count for YORK: " + str(uni['YORK']))
print("bigram count for NEW YORK: " + str(bi[('NEW', 'YORK')]))


# Data for plotting
s = uni.values()
t = []
for value in s:
    t.append(value)
#print(t)
t.sort()
#print(t)
#print(np.log(s))
fig, ax = plt.subplots()

ax.plot(t)
plt.yscale('log')
plt.xscale('log')
#plt.show()
fig.savefig("test.png")

def bigram_threshold(bigram):
    new_bigrams = defaultdict(float)
    for key in bigram:
        if bigram[key] >= 100:
            new_bigrams[key] = bigram[key]
    return new_bigrams

hundred_bi = bigram_threshold(bi)
all_pmi_hundred = all_pmi(uni, hundred_bi)

sorted_pmi_hundred = sorted(all_pmi_hundred, key=all_pmi_hundred.get, reverse=True)
top_thirty_pmi_hundred = sorted_pmi_hundred[:10]
print("top thirty pmi pairs hundred: " + str(top_thirty_pmi_hundred))


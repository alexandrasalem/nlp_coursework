from hw2_utils.constants import OFFSET
from hw2_utils import clf_base, evaluation

import numpy as np
from collections import defaultdict, Counter
from itertools import chain
import math

# deliverable 3.1
def get_corpus_counts(x,y,label):
    """
    Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    #raise NotImplementedError
    #x is list of counters
    #y is list of labels

    corpus_counts = defaultdict(float)
    for i in range(0, len(y)):
        if y[i] == label:
            #print(y[i])
            for word in x[i]:
                corpus_counts[word] += x[i][word]

    return corpus_counts

# deliverable 3.2
def estimate_pxy(x,y,label,smoothing,vocab):
    """
    Compute smoothed log-probability P(word | label) for a given label. (eq. 2.30 in Eisenstein, 4.14 in J&M)

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    """
    #raise NotImplementedError
    #this is the laplace smoothing I have implemented before
    #but now, instead of with a language model, it's for a classifier

    dict_of_probs = defaultdict(float)

    corpus_counts = get_corpus_counts(x,y,label)

    count_class = 0.0
    for word in corpus_counts:
        count_class+=corpus_counts[word]

    for word in vocab:
        prob = np.log((corpus_counts[word] + smoothing)/((len(vocab)*smoothing) + count_class))
        dict_of_probs[word] = prob

    return dict_of_probs

# deliverable 3.3
def estimate_nb(x,y,smoothing):
    """
    Estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights, as a default dict where the keys are (label, word) tuples and values are smoothed log-probs of P(word|label)
    :rtype: defaultdict 

    """
    
    labels = set(y)
    counts = defaultdict(float)
    doc_counts = defaultdict(float)
    
    #raise NotImplementedError
    #make vocab from x

    vocab = set()
    for counter in x:
        for key in counter:
            vocab.add(key)
    #vocab.add(OFFSET)

    vocab = list(vocab)

    theta = defaultdict(float)
    for label in labels:
        probs_label = estimate_pxy(x, y, label, smoothing, vocab)
        #print(probs_label)
        for word in vocab:
        #for key in probs_label:
            theta[(label, word)] = probs_label[word]
        #theta[(label, OFFSET)] = (1/4)

    return theta

    

# deliverable 3.4
def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    """
    Find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values
    :returns: best smoothing value, scores
    :rtype: float, dict mapping smoothing value to score
    """

    #raise NotImplementedError
    #best_accuracy = 0.0
    #best_smoother = smoothers[0]
    best_dict = dict()
    for smoother in smoothers:
        print(smoother)
        theta_nb = estimate_nb(x_tr, y_tr, smoother)
        print("made theta_nb")
        y_ht = clf_base.predict_all(x_dv, theta_nb, set(y_tr))
        print("made y_hat")
        accuracy = evaluation.acc(y_ht, y_dv)
        print(accuracy)
        best_dict[smoother] = accuracy


        #if accuracy > best_accuracy:
        #    best_accuracy = accuracy
        #    best_smoother = smoother
        #    best_dict = y_ht
    best_score = clf_base.argmax(best_dict)
    return best_score, best_dict


    
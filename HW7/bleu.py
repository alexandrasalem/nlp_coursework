#!/usr/bin/env python -O
# basic BLEU score implementation
# Kyle Gorman <gormanky@ohsu.edu>

from __future__ import division

from math import exp, log
from collections import Counter


MAX_GRAM_SIZE = 4



# helpers

def ngrams(tokens, order):
    """
    Generate n-grams of order `order`, with no padding
    """ 
    for i in xrange(len(tokens) - order + 1):
        yield tuple(tokens[i:i + order])

def bitext(source, target):
    """
    Run through the bitext files, yielding one sentence at a time
    """
    for (s, t) in zip(source, target):
        yield (s.strip().split(), t.strip().split())
        # this time, we don't want a null symbol


def BLEU(hypothesis, reference, n=MAX_GRAM_SIZE):
    """
    Compute BLEU for a hypothesis/reference sentence pair
    """
    #raise NotImplementedError
    log_bleu = 0.0
    brev = min((1-len(hypothesis)/len(reference)), 0)
    for n_gram in range(1, n+1):
        cand_n_grams = ngrams(hypothesis, n_gram)
        ref_n_grams = ngrams(reference, n_gram)
        cand_counter = Counter()
        ref_counter = Counter()
        for ngram in cand_n_grams:
            cand_counter.update([ngram])
        for ngram in ref_n_grams:
            ref_counter.update([ngram])
            
        cand_n_grams = ngrams(hypothesis, n_gram)
        ref_n_grams = ngrams(reference, n_gram)
        p_n = 0.0
        for ngram in cand_n_grams:
            #print(cand_counter(ngram))
            #print(ref_counter(ngram))
            count_clipped_ngram = min(cand_counter[ngram], ref_counter[ngram])
            p_n += count_clipped_ngram
        
        total = sum(ref_counter.values())
        p_n = p_n/total
        
        #print(p_n)
        if p_n == 0:
            bleu = 0
            return bleu
        else:
            new_log_bleu = (1/n_gram) * log(p_n)
        log_bleu += new_log_bleu
        
    log_bleu = log_bleu + brev
    bleu = exp(log_bleu)
    return bleu

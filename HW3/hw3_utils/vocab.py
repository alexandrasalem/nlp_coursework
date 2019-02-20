from collections import defaultdict
from itertools import count
import torch

BOS_SYM = '<BOS>'
EOS_SYM = '<EOS>'

def build_vocab(corpus):
    """
    Build an exhaustive character inventory for the corpus, and return dictionaries
    that can be used to map characters to indicies and vice-versa.
    
    Make sure to include BOS_SYM and EOS_SYM!
    
    :param corpus: a corpus, represented as an iterable of strings
    :returns: two dictionaries, one mapping characters to vocab indicies and another mapping idicies to characters.
    :rtype: dict-like object, dict-like object
    """
    #raise NotImplementedError
    
    char_to_indicies = dict()
    indicies_to_char = dict()
    
    vocab_set = set()
    
    for string in corpus:
        string_list = list(string)
        vocab_set.update(string_list)
        
    
    char_to_indicies[BOS_SYM] = 0
    indicies_to_char[0] = BOS_SYM
    
    char_to_indicies[EOS_SYM] = 1
    indicies_to_char[1] = EOS_SYM
    
    i = 2
    for char in vocab_set:
        char_to_indicies[char] = i
        indicies_to_char[i] = char
        i+=1
    
    return char_to_indicies, indicies_to_char
    
def sentence_to_vector(s, vocab, pad_with_bos=False):
    """
    Turn a string, s, into a list of indicies in from `vocab`. 
    
    :param s: A string to turn into a vector
    :param vocab: the mapping from characters to indicies
    :param pad_with_bos: Pad the sentence with BOS_SYM/EOS_SYM markers
    :returns: a list of the character indicies found in `s`
    :rtype: list
    """
    #raise NotImplementedError
    
    
    s_vector = list(s)
    indicies_vector = []
    if pad_with_bos:
        indicies_vector.append(vocab[BOS_SYM])
    for char in s_vector:
        indicies_vector.append(vocab[char])
    if pad_with_bos:
        indicies_vector.append(vocab[EOS_SYM])
        
    return indicies_vector
    
    
def sentence_to_tensor(s, vocab, pad_with_bos=False):
    """
    :param s: A string to turn into a tensor
    :param vocab: the mapping from characters to indicies
    :param pad_with_bos: Pad the sentence with BOS_SYM/EOS_SYM markers
    :returns: (1, n) tensor where n=len(s) and the values are character indicies
    :rtype: torch.Tensor
    """
    #raise NotImplementedError
    
    sentence_vector = sentence_to_vector(s, vocab, pad_with_bos)
    sentence_tensor = torch.Tensor([sentence_vector])
    return sentence_tensor
    
def build_label_vocab(labels):
    """
    Similar to build_vocab()- take a list of observed labels and return a pair of mappings to go from label to numeric index and back.
    
    The number of label indicies should be equal to the number of *distinct* labels that we see in our dataset.
    
    :param labels: a list of observed labels ("y" values)
    :returns: two dictionaries, one mapping label to indicies and the other mapping indicies to label
    :rtype: dict-like object, dict-like object
    """
    #raise NotImplementedError
    
    labels_to_indicies = dict()
    indicies_to_labels = dict()
    
    label_set = set(labels)       
    
    i = 0
    for label in label_set:
        labels_to_indicies[label] = i
        indicies_to_labels[i] = label
        i+=1
    
    return labels_to_indicies, indicies_to_labels
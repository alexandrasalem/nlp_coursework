from  hw2_utils import constants 
import numpy as np

# deliverable 4.1
def get_token_type_ratio(counts):
    """
    Compute the ratio of tokens to types
    
    :param counts: bag of words feature for a song
    :returns: ratio of tokens to types
    :rtype float
    """
    #raise NotImplementedError
    length_of_song_in_tokens = sum(counts.values())
    number_of_distinct_types = len(counts.keys())
    if length_of_song_in_tokens == 0 or number_of_distinct_types == 0:
        token_type_ratio = 0
    else:
        token_type_ratio = length_of_song_in_tokens/number_of_distinct_types
    return token_type_ratio


# deliverable 4.2
def concat_ttr_binned_features(data):
    """
    Add binned token-type ratio features to the observation represented by data
    
    :param data: Bag of words
    :returns: Bag of words, plus binned ttr features
    :rtype: dict
    """
    #raise NotImplementedError
    data_plus_ttr = data
    token_type_ratio = get_token_type_ratio(data)

    ttrs = [constants.TTR_ZERO, constants.TTR_ONE, constants.TTR_TWO, constants.TTR_THREE, constants.TTR_FOUR, constants.TTR_FIVE, constants.TTR_SIX]
    for ttr in ttrs:
        data_plus_ttr[ttr] = 0

    if 0<token_type_ratio<1:
        data_plus_ttr[constants.TTR_ZERO] = 1
    elif 1<=token_type_ratio<2:
        data_plus_ttr[constants.TTR_ONE] = 1
    elif 2<= token_type_ratio <3:
        data_plus_ttr[constants.TTR_TWO] = 1
    elif 3<=token_type_ratio <4:
        data_plus_ttr[constants.TTR_THREE] = 1
    elif 4<=token_type_ratio<5:
        data_plus_ttr[constants.TTR_FOUR] = 1
    elif 5<=token_type_ratio<6:
        data_plus_ttr[constants.TTR_FIVE] = 1
    else:
        data_plus_ttr[constants.TTR_SIX] = 1

    return data_plus_ttr

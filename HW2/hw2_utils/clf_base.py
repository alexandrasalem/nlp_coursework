from hw2_utils.constants import OFFSET
import numpy as np
import operator

# HELPER FUNCTION
def argmax(score_dict):
    """
    Find the 
    :param score_dict: A dict whose keys are labels and values are scores
    :returns: Top-scoring label
    :rtype: string
    """
    items = list(score_dict.items())
    items.sort()
    return items[np.argmax([i[1] for i in items])][0]


# deliverable 2.1
def make_feature_vector(base_features, label):
    """
    Take a dict of bag-of-words features and a label; return a dict of features, corresponding to f(x,y)
    
    :param base_features: Counter of base features
    :param label: label string
    :returns dict of features, f(x,y)
    :rtype: dict
    """
    #raise NotImplementedError
    feature_vector = dict()
    feature_vector[(label, OFFSET)] = 1
    for key in base_features:
        new_key = (label, key)
        feature_vector[new_key] = base_features[key]
    return feature_vector
    
# deliverable 2.2
def predict(base_features, weights, labels):
    """
    Simple linear prediction function y_hat = argmax_y \theta^T f(x,y)
    
    :param base_features: a dictionary of base features and counts (base features, NOT a full feature vector)
    :param weights: a defaultdict of features and weights. Features are tuples (label, base_feature)
    :param labels: a list of candidate labels
    :returns: top-scoring label, plus the scores of all labels
    :rtype: string, dict
    """
    #raise NotImplementedError

    #print(base_features)

    #feature_vector = make_feature_vector(base_features)
    scores_dict = dict()
    for label in labels:
        scores_dict[label] = float(0)

    for label in labels:
        feature_vector = make_feature_vector(base_features, label)
        for key in weights:
            if key in feature_vector:
                scores_dict[label] += float(float(feature_vector[key]) * float(weights[key]))

    #scores_dict = dict()
    #for label in labels:
    #    scores_dict[label] = float(0)

    #for key in weights:
    #    word = key[1]
    #    label = key[0]
    #    if word == OFFSET:
    #        scores_dict[label] = weights[key]
    #        continue
    #    scores_dict[label] += (float(base_features[word])*weights[key])


    #max(scores_dict.items(), key=operator.itemgetter(1))[0]
    
    max_key = argmax(scores_dict)
    #max_key = max(scores_dict, key=scores_dict.get)
    return (max_key, scores_dict)
    
def predict_all(x, weights, labels):
    """
    Predict the label for all instances in a dataset. For bulk prediction.
    
    :param x: iterable of base instances
    :param weights: defaultdict of weights
    :param labels: a list of candidate labels
    :returns: predictions for each instance
    :rtype: numpy array
    """
    y_hat = np.array([predict(x_i, weights, labels)[0] for x_i in x])
    return y_hat
    
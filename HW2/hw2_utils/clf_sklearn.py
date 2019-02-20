from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer

# deliverable 5.1
def train_logistic_regression(X, y):
    """
    Train a Logistic Regression classifier
    
    Pay attention to the mult_class parameter to control how the classifier handles multinomial situations!
    
    :params X: a sparse matrix of features
    :params y: a list of instance labels
    :returns: a trained logistic regression classifier
    :rtype sklearn.linear_model.LogisticRegression
    """
    #raise NotImplementedError
    test = LogisticRegression()
    test = test.fit(X, y)
    return test

def transform_tf_idf(X_train_counts, X_dev_counts, X_test_counts):
    """
    :params X_train_counts: the bag-of-words matrix producd by CountVectorizer for the training split
    :params X_dev_counts: the same, but for the dev split
    :params X_test_counts: ditto, for the test split
    :returns: a tuple of tf-idf transformed count matrices for train/dev/test (in that order), as well as the resulting transformer
    :rtype ((sparse, sparse, sparse), TfidfTransformer)
    """
    #raise NotImplementedError
    #print(X_train_counts.shape)
    tfidf_trans = TfidfTransformer(use_idf=True).fit(X_train_counts)
    X_train_tfidf = tfidf_trans.transform(X_train_counts)

    tfidf_trans = tfidf_trans.fit(X_dev_counts)
    X_dev_tfidf = tfidf_trans.transform(X_dev_counts)

    tfidf_trans = tfidf_trans.fit(X_test_counts)
    X_test_tfidf = tfidf_trans.transform(X_test_counts)

    #tf_trans = TfidfTransformer(use_idf=True).fit(X_test_counts)
    #tf_X_test = tf_trans.fit_transform(X_test_counts)

    return ((X_train_tfidf, X_dev_tfidf, X_test_tfidf), tfidf_trans)
    
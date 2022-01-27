from distutils.errors import DistutilsPlatformError
from doctest import DocFileSuite
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sympy import Complement


PREDICTORS = ['token_no_stop','lemma','pos','prev_lemma','next_lemma','prev_pos','next_pos','snowball_stemmer',
            'porter_stemmer','head','dependency','is_part_of_negation','has_prefix','has_postfix','has_infix',
            'base_in_dictionary','has_apostrophe']


def format_data(df):

    # prediction label to last column
    new_cols = [col for col in df.columns if col != 'label'] + ['label']
    df = df[new_cols]

    # fill NaN with 'no_label'
    df = df.fillna('no_label')

    # true/false to 1/0
    df["is_part_of_negation"] = df["is_part_of_negation"].astype(int)
    df["has_prefix"] = df["has_prefix"].astype(int)
    df["has_postfix"] = df["has_postfix"].astype(int)
    df["has_infix"] = df["has_infix"].astype(int)
    df["base_in_dictionary"] = df["base_in_dictionary"].astype(int)
    df["has_apostrophe"] = df["has_apostrophe"].astype(int)

    # print(df.columns)
    # print(df.isna().any())

    return df


def vectorize_split_data(df_train, df_val):

    dict_vec = DictVectorizer(sparse=False)

    X_train = dict_vec.fit_transform(df_train[PREDICTORS].to_dict('records'))
    y_train = df_train.iloc[:, -1].to_numpy()

    X_val = dict_vec.transform(df_val[PREDICTORS].to_dict('records'))
    y_val = df_val.iloc[:, -1].to_numpy()

    print(type(X_train))
    print(type(y_train))

    # print(len(x))
    # print(y.shape())

    return X_train, y_train, X_val, y_val


def run_naive_bayes(X_train, y_train, X_val, y_val):

    # multinomial nb
    clf = MultinomialNB()
    # clf = ComplementNB()
    print('Fitting the model...')
    clf.fit(X_train, y_train)
    print('Predicting...')
    predictions = clf.predict(X_val)

    return clf, predictions


def evaluation(clf, X_val, y_val, predictions):

    clf_report = pd.DataFrame(classification_report(y_true = y_val, y_pred = predictions, output_dict=True)).transpose()
    print(clf_report)

    plot_confusion_matrix(clf, X_val, y_val)  
    plt.show()

    # confusion_matrix = pd.crosstab(df_val['label'], df_val['prediction'], rownames=['Actual'], colnames=['Predicted'])
    # sn.heatmap(confusion_matrix, annot=True, cmap='Blues')
    # plt.show()

    return


def main():

    df_train = pd.read_csv('SEM2012_training_data_with_features.csv')
    df_val = pd.read_csv('SEM2012_validation_data_with_features.csv')

    df_train = format_data(df_train)
    df_val = format_data(df_val)

    X_train, y_train, X_val, y_val = vectorize_split_data(df_train, df_val)

    clf, predictions = run_naive_bayes(X_train, y_train, X_val, y_val)
    df_val['prediction'] = predictions

    evaluation(clf, X_val, y_val, predictions)


if __name__ == '__main__':
    main()




















# # https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
# def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
#     # fit the training dataset on the classifier
#     classifier.fit(feature_vector_train, label)
    
#     # predict the labels on validation dataset
#     predictions = classifier.predict(feature_vector_valid)
    
#     if is_neural_net:
#         predictions = predictions.argmax(axis=-1)
    
#     return metrics.accuracy_score(predictions, valid_y)


#     # Naive Bayes on Count Vectors
# accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
# print "NB, Count Vectors: ", accuracy

# # Naive Bayes on Word Level TF IDF Vectors
# accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
# print "NB, WordLevel TF-IDF: ", accuracy

# # Naive Bayes on Ngram Level TF IDF Vectors
# accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
# print "NB, N-Gram Vectors: ", accuracy

# # Naive Bayes on Character Level TF IDF Vectors
# accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
# print "NB, CharLevel Vectors: ", accuracy
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import classification_report, confusion_matrix



df_train = pd.read_csv('SEM2012_training_data_with_features.csv')
df_val = pd.read_csv('SEM2012_validation_data_with_features.csv')

# prediction label to last column
new_cols = [col for col in df_train.columns if col != 'label'] + ['label']
df_train = df_train[new_cols]
df_val = df_val[new_cols]

# true/false to 1/0
df_train["is_part_of_negation"] = df_train["is_part_of_negation"].astype(int)
df_train["has_prefix"] = df_train["has_prefix"].astype(int)
df_train["has_postfix"] = df_train["has_postfix"].astype(int)
df_train["has_infix"] = df_train["has_infix"].astype(int)
df_train["base_in_dictionary"] = df_train["base_in_dictionary"].astype(int)
df_train["has_apostrophe"] = df_train["has_apostrophe"].astype(int)

# vectorize strings
dict_vec = DictVectorizer()
dict_vec.fit(df_train[['token_no_stop']].to_dict('records'))
X_train = dict_vec.transform(df_train[['token_no_stop']].to_dict('records'))
X_val = dict_vec.transform(df_val[['token_no_stop']].to_dict('records'))

print(X_train)


# count_vec = CountVectorizer()
# count_vec.fit(df_train['token_no_stop'])
# X_train = count_vec.transform(df_train['token_no_stop'])
# X_val = count_vec.transform(df_val['token_no_stop'])



# split data in X and y
# X_train = df_train.iloc[:, np.r_[18:21,23:24]].to_numpy()
y_train = df_train.iloc[:, -1].to_numpy()
# X_val = df_val.iloc[:, np.r_[18:21,23:24]].to_numpy()    
y_val = df_val.iloc[:, -1].to_numpy()


# naive bayes classification (multinomial and complement)
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
predictions = mnb.predict(X_val)

# cnb = ComplementNB()
# cnb.fit(X_train, y_train)
# predictions = cnb.predict(X_val)


# visualisation
df_val['prediction'] = predictions
clsf_report = pd.DataFrame(classification_report(y_true = df_val['label'], y_pred = df_val['prediction'], output_dict=True)).transpose()
print(clsf_report)

confusion_matrix = pd.crosstab(df_val['label'], df_val['prediction'], rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True, cmap='Blues')
plt.show()






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
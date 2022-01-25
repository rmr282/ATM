import numpy as np
import pandas as pd

df_train = pd.read_csv('SEM2012_training_data_with_features.csv')
df_val = pd.read_csv('SEM2012_validation_data_with_features.csv')

negations = df_train.loc[df_train['label'].isin(['B-NEG', 'I-NEG'])]['token_no_stop'].to_list()
negations = set(negations)
negations.remove('the', 'at', 'for', 'by', 'all', 'means')
print(negations)



predictions = []

for index, token in df_val['token_no_stop'].iteritems():

    if token in negations:
        predictions.append('B-NEG')
        print(token)
    else:
        predictions.append('O')




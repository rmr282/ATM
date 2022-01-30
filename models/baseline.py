import numpy as np
import pandas as pd
import regex as re
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


df_train = pd.read_csv('SEM2012_training_data_with_features.csv')
df_val = pd.read_csv('SEM2012_validation_data_with_features.csv')

# negations = df_train.loc[df_train['label'].isin(['B-NEG', 'I-NEG'])]['token_no_stop'].to_list()
# negations = sorted(set(negations))
# negations.remove('the', 'at', 'for', 'by', 'all', 'means')
# print(negations)

# ['absence', 'all', 'at', 'breathless', 'breathlessness', 'by', 'careless', 'carelessness', 'colourless', 'contrary', 'disapprobation', 'disconnected', 'disfavour', 
# 'displeasure', 'distasteful', 'except', 'fail', 'for', 'godless', 'harmless', 'helpless', 'helplessly', 'hopeless', 'immaterial', 'immutable', 'impassable', 'impatient', 
# 'impatiently', 'impenetrable', 'impossible', 'imprudent', 'inadequate', 'inadvertently', 'inconceivable', 'inconclusive', 'inconvenient', 'incredible', 'incredulously', 
# 'indescribably', 'indiscreet', 'inexplicable', 'infinite', 'infrequent', 'inhospitable', 'inscrutable', 'insensible', 'interminable', 'intolerable', 'invisible', 
# 'irregular', 'irrelevant', 'irresolute', 'irretrievably', 'irrevocable', 'lifeless', 'means', 'motionless', "n't", 'neglected', 'neither', 'never', 'no', 'nobody', 
# 'noiselessly', 'none', 'nor', 'nothing', 'nowhere', 'on', 'powerless', 'prevent', 'purposeless', 'rather', 'refused', 'restlessly', 'save', 'shelterless', 'than', 
# 'the', 'unable', 'unambitious', 'unarmed', 'unbroken', 'uncanny', 'uncertain', 'uncomfortably', 'uncommon', 'unconcerned', 'unconcernedly', 'unconscious', 
# 'uncontrollable', 'uncurtained', 'undeceived', 'undeniable', 'uneasiness', 'uneasy', 'uneducated', 'unemotional', 'unexpected', 'unexplored', 'unfair', 'unfairly', 
# 'unfortunate', 'unfortunately', 'unfounded', 'unfruitful', 'unfurnished', 'unhappy', 'unimaginative', 'uninteresting', 'unjustifiable', 'unknown', 'unlike', 
# 'unlikely', 'unlimited', 'unmarried', 'unmistakable', 'unmitigated', 'unnatural', 'unnecessary', 'unoccupied', 'unpleasant', 'unpractical', 'unsafe', 'unseen', 
# 'unsigned', 'untenanted', 'untimely', 'unusual', 'unwarlike', 'useless', 'windless', 'without', 'world', 'worthless']


negations = ('neither', 'never', 'no', 'nobody', 'none', 'nor', 'nothing', 'nowhere', 'except', 'rather', 'than', 'without')
prefixes = ('in', 'un', 'non', 'de', 'dis', 'a', 'anti', 'im', 'il', 'ir')
postfixes = ('less')

predictions = []

for index, token in df_val['token_no_stop'].iteritems():

    negation_boolean = token.startswith(prefixes)
    prefix_boolean = token.startswith(prefixes)
    postfix_boolean = token.endswith(postfixes)
    infix_match = re.search(r'\wless\w', token)
    apostrophe_match = re.search(r'\w\'\w', token) 

    if negation_boolean == True:
        predictions.append('B-NEG')
    elif prefix_boolean == True:
        predictions.append('B-NEG')
    elif postfix_boolean == True:
        predictions.append('B-NEG')
    elif infix_match is not None:
        predictions.append('B-NEG')
    elif apostrophe_match is not None:
        predictions.append('B-NEG')
    else:
        predictions.append('O')


df_val['prediction'] = predictions
# df_val[['prediction', 'label']].to_csv('baseline_predictions.csv', index=False)

# visualisation
clsf_report = pd.DataFrame(classification_report(y_true = df_val['label'], y_pred = df_val['prediction'], output_dict=True)).transpose()
print(clsf_report)

confusion_matrix = pd.crosstab(df_val['label'], df_val['prediction'], rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True, cmap='Blues')
plt.show()
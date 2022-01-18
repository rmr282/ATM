import os
import pandas as pd 
import regex as re
# import spacy


SEM_LOC = os.getcwd() + '\\SEM-2012-SharedTask-CD-SCO-simple.v2\\'

col_names = ['annotater', 'sentence_id', 'token_id', 'token', 'negation']
df_training = pd.read_csv(SEM_LOC + 'SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt', sep = '\t', header = None, names=col_names)

# DATA EXPLORATION

print(df_training.count())
print(df_training.head(10))
print(df_training.nunique())
print(df_training['negation'].value_counts())
print(df_training['token'].value_counts())
print(df_training[df_training['negation'] == 'B-NEG']['token'].value_counts())

# PREPROCESSING

df_training['token'] = df_training['token'].str.lower()
print(df_training.head(10))
print(df_training[df_training['negation'] == 'B-NEG']['token'].value_counts().head(20))

#TODO: graph

# FEATURE ENGINGEERING

prefixes = ['in', 'un', 'non', 'de', 'dis', 'a', 'anti', 'im', 'il', 'ir']
postfixes = ['less'] # also use for infix

# print re.findall(r"(?=("+'|'.join(string_lst)+r"))", x)

# df_training['has_prefix'] = re.search('in', df_training['token'].str)

has_prefix = []
has_postfix = []
has_infix = []
has_apostrophe = []
multi_word = []

for index, row in enumerate(df_training.values): 
    token = row[3]
    prefix = re.search(r'^(?=('+'|'.join(prefixes)+r'))', token)
    postfix = re.search('less$', token)
    infix = re.search(r'\wless\w', token) # r'[^less ]\w*less\w*[^less ]'
    apostrophe = re.search(r'\w\'\w', token)

    if prefix is not None:
        # print(prefix.group(0))
        has_prefix.append(1)
    else:
        has_prefix.append(0)

    if postfix is not None:
        # print(postfix.group(0))
        has_postfix.append(1)
    else:
        has_postfix.append(0)

    if infix is not None:
        # print(infix.group(0))
        has_infix.append(1)
    else:
        has_infix.append(0)

    if apostrophe is not None:
        # print(apostrophe.group(0))
        has_apostrophe.append(1)
    else:
        has_apostrophe.append(0)


df_training['has_prefix'] = has_prefix
df_training['has_postfix'] = has_postfix
df_training['has_infix'] = has_infix
df_training['has_apostrophe'] = has_apostrophe

print(df_training[df_training['negation'] == 'B-NEG'])

print(df_training[df_training['has_apostrophe'] == 1].head(30))

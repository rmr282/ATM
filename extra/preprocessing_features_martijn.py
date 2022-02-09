import os
import pandas as pd 
import regex as re
import nltk
import copy

# nltk.download('words')
english_vocab = set(w.lower() for w in nltk.corpus.words.words())

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
postfixes = ('less') # also use for infix

has_prefix = []
has_postfix = []
has_infix = []
has_apostrophe = []
base_tokens = []
bases_in_dict = []

multi_word = [] # by no means, not for the world, on the contrary, nothing at all, rather than


for index, row in enumerate(df_training.values): 
    
    token = row[3]
    
    prefix_match = re.search(r'^(?=('+'|'.join(prefixes)+r'))', token) #token.startswith(prefixes)
    postfix_match = re.search('less$', token) #token.endswith(postfixes)
    infix_match = re.search(r'\wless\w', token) # r'[^less ]\w*less\w*[^less ]'
    apostrophe_match = re.search(r'\w\'\w', token) # or "n't"

    base_token = copy.deepcopy(token)
    base_in_dict = base_token in english_vocab

    # prefix
    if (prefix_match is not None) and (len(token) >= 5):
        if prefix_match.groups(0):
            prefix = prefix_match.groups(0)[0]
            base_token = token.replace(prefix, '')
            base_in_dict = base_token in english_vocab
            has_prefix.append(True)
        else:
            has_prefix.append(False)
    else:
        has_prefix.append(False)

    # postfix
    if (postfix_match is not None) and (len(token) >= 5):
        if postfix_match.group(0):
            postfix = postfix_match.group(0)
            base_token = token.replace(postfix, '')
            base_in_dict = base_token in english_vocab
            has_postfix.append(True)
        else: 
            has_postfix.append(False)    
    else:
        has_postfix.append(False)

    # infix
    if infix_match is not None:
        has_infix.append(True)
    else:
        has_infix.append(False)

    # apostrophe
    if apostrophe_match is not None:
        has_apostrophe.append(True)
    else:
        has_apostrophe.append(False)

    base_tokens.append(base_token)
    bases_in_dict.append(base_in_dict)
    

df_training['has_prefix'] = has_prefix
df_training['has_postfix'] = has_postfix
df_training['has_infix'] = has_infix
df_training['base'] = base_tokens
df_training['base_in_dictionary'] = bases_in_dict
df_training['has_apostrophe'] = has_apostrophe

print(df_training[df_training['negation'] == 'I-NEG'])

print(df_training[df_training['has_postfix'] == True].head(30))

# print(df_training[(df_training['sentence_id'] == 200) & (df_training['annotater'] == 'baskervilles09')].head(30))


import os
import copy
from collections import Counter
from collections import defaultdict
import regex as re
import string
import pandas as pd 
import numpy as np

import spacy

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from stanza.pipeline.processor import ProcessorVariant, register_processor_variant



# --- DEFINITIONS -------------------------------------------------------------------------------------

#TODO: make this more organized

# nltk.download('words')
english_vocab = set(w.lower() for w in nltk.corpus.words.words())

nlp = spacy.load("en_core_web_sm")

SEM_LOC = os.getcwd() + '\\SEM-2012-SharedTask-CD-SCO-simple.v2\\'


# define punctuations
punctuations = string.punctuation
punctuations = punctuations.replace("'", '')
punctuations = punctuations.replace('`', '')

# define stopwords
all_stopwords = stopwords.words('english')
adjusted_stopwords = [e for e in all_stopwords if e not in ('ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",
'don', "don't", 'should', 't', 'can', 'no', 'nor', 'not', 'only', 'do', 'does', 'are', 'was', 'were', 'have', 'has', 'had', 'against', 'by', 'the', 'for')]




# --- DATA EXPLORATION -----------------------------------------------------------------------------------

#TODO: graph

def exploration(df_training): 
    print(df_training.count())
    print(df_training.head(10))
    print(df_training.nunique())
    print(df_training['label'].value_counts())
    print(df_training['token'].value_counts())
    print(df_training[df_training['label'] == 'B-NEG']['token'].value_counts())


def get_statistics_ravi(data):

    annotator_ids = data["annotator"].unique()
    TOTAL_SENTENCES = []
    # print(annotator_ids)
    for annotator in annotator_ids:
        temp = data.loc[data["annotator"]==annotator]
        list_sentences = temp['sentence_id'].unique()
        # print(list_sentences)
        for sentence in list_sentences:
            temp2 = temp.loc[temp['sentence_id'] == sentence]
            tokens = ' '.join(list(temp2['token']))
            ## REMOVE PUNCTUATION EXCEPT APOSTROPHE
            tokens = " ".join("".join([" " if ch in punctuations else ch for ch in tokens]).split())
            TOTAL_SENTENCES.append(tokens)

    print('Number of Sentences:', len(TOTAL_SENTENCES))
    print('Number of tokens:', len(data['token']))

    # Counting the frequencies of the words 
    word_frequencies = Counter()
    total_length = 0

    # To count the number of tokens 
    num_tokens = 0
    pos_tokencounts_dict = defaultdict(Counter)

    annotator_ids = data["annotator"].unique()

    for annotator in annotator_ids:
        temp = data.loc[data['annotator'] == annotator]
        list_sentences = pd.unique(data[['sentence_id']].values.ravel())

        for sentence in list_sentences:
            words = []
            pos_list = []
            tag_list = []
            temp2 = temp.loc[temp['sentence_id'] == sentence]

            tokens = ' '.join(list(temp2['token']))
            tokens = " ".join("".join([" " if ch in punctuations else ch for ch in tokens]).split())
            doc = nlp(tokens)
            prev = None
            prev_lemma = []
            lemmas = []
            for token in doc:
                num_tokens += 1
                prev_lemma = prev
                lemmas.append(token.lemma_)
                if not token.is_punct:
                    total_length += len(token)
                    tags_tp = (token.tag_,token.pos_)
                    pos_tokencounts_dict[tags_tp].update([token.text])
                    words.append(token.text)
            word_frequencies.update(words)

    # Number of words 
    num_words = sum(word_frequencies.values())
    print('Number of words:', num_words)

    # Number of word types 
    num_types = len(word_frequencies.keys())
    print('Number of word types:', num_types)
    # Average words per sentence 

    avg_words_sen = num_words/len(TOTAL_SENTENCES)
    print('Average words per sentence:', avg_words_sen)

    # Average word length 
    avg_word_length = total_length / num_words
    print('Average word length', avg_word_length)


    pos_tokencounts_dict


    return



# --- PREPROCESSING ---------------------------------------------------------------------------------------

def remove_punctuations(text):
    for punctuation in punctuations:
        text = text.replace(punctuation, '')
    
    return text


def preproccessing(data):

    # apply lowering, removing punctuations and deleting stopwords 
    data['token_lower'] = data['token'].str.lower()
    data['token_no_punct'] = data['token_lower'].apply(remove_punctuations)
    data['token_no_stop'] = data['token_no_punct'].apply(lambda x: ' '.join([word for word in x.split() if word not in (adjusted_stopwords)]))

    # remove examples with empty token 
    data['token_no_stop'].replace('', np.nan, inplace=True)
    data.dropna(subset=['token_no_stop'], inplace=True)

    return data



# --- FEATURE ENGINGEERING ----------------------------------------------------------------------------------

def feature_extraction(data):




    return data


def features_ravi(data):

    spacy_pipe = nlp.pipe(data["token_lower"].values, disable=["ner", "parser"])
    # get lemma, pos, previous lemma, previous pos, next lemma, next pos 
    features_gen = ((doc[0].lemma_, doc[0].pos_) for doc in spacy_pipe)
    data["lemma"], data["pos"] = zip(*features_gen)
    data["prev_Lemma"] = data["lemma"].shift(periods=1)
    data["next_Lemma"] = data["lemma"].shift(periods=-1)
    data["prev_pos"] = data["pos"].shift(periods=1)
    data["next_pos"] = data["pos"].shift(periods=-1)

    # trying some things out with stemmers 
    snow = SnowballStemmer(language='english')
    data["snowballStemmer"] = data.apply(lambda row: snow.stem(row["token_lower"]), axis=1)
    port = PorterStemmer()
    data['PorterStemmer'] = data.apply(lambda row: port.stem(row["token_lower"]), axis=1)


    return data


def get_regex_features(df_training):

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


    return df_training


# --- MAIN --------------------------------------------------------------------------------------------------

def main():

    data = pd.read_csv(SEM_LOC + 'SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt', sep="\t", header=None)
    data.columns = ['annotator', 'sentence_id', 'token_id', 'token', 'label']


    df_training = get


    print(df_training[df_training['label'] == 'I-NEG'])

    print(df_training[df_training['has_postfix'] == True].head(30))

    # print(df_training[(df_training['sentence_id'] == 200) & (df_training['annotater'] == 'baskervilles09')].head(30))


if __name__ == '__main__':
    main()
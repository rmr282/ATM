import os
import copy
from collections import Counter
from collections import defaultdict

import regex as re
import string
import pandas as pd 
import numpy as np

import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.parse import CoreNLPParser
from nltk.parse.corenlp import CoreNLPDependencyParser

#TODO: uncomment when running for first time
# nltk.download('words') 

# from stanza.pipeline.processor import ProcessorVariant, register_processor_variant



# --- DEFINITIONS -------------------------------------------------------------------------------------

DATA_DIR = os.getcwd() + '\\SEM-2012-SharedTask-CD-SCO-simple.v2\\'

english_vocab = set(w.lower() for w in nltk.corpus.words.words())
nlp = spacy.load("en_core_web_sm")

# define punctuations
punctuations = string.punctuation
punctuations = punctuations.replace("'", '')
punctuations = punctuations.replace('`', '')

# define stopwords
all_stopwords = stopwords.words('english')
adjusted_stopwords = [e for e in all_stopwords if e not in ('ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', 
"hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'don', "don't", 'should', 't', 'can', 'no', 'nor', 
'not', 'only', 'do', 'does', 'are', 'was', 'were', 'have', 'has', 'had', 'against', 'by', 'the', 'for')]




# --- DATA EXPLORATION -----------------------------------------------------------------------------------

#TODO: graph

def exploration(data): 
    print('Unique values per column')
    print(data.nunique())
    print('Number of instance labels')
    print(data['label'].value_counts())
    print('Most frequent tokens')
    print(data['token'].value_counts().head(10))
    print('Most frequent negation cues')
    print(data[data['label'] == 'B-NEG']['token'].value_counts().head(10))


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



# --- PREPROCESSING ---------------------------------------------------------------------------------------

def remove_punctuations(text):   
    for punctuation in punctuations:
        text = text.replace(punctuation, '')
    
    return text


def preprocessing(data):
    # apply lowering, removing punctuations and deleting stopwords 
    data['token_lower'] = data['token'].str.lower()
    data['token_no_punct'] = data['token_lower'].apply(remove_punctuations)
    data['token_no_stop'] = data['token_no_punct'].apply(lambda x: ' '.join([word for word in x.split() if word not in (adjusted_stopwords)]))

    # remove examples with empty token 
    data['token_no_stop'].replace('', np.nan, inplace=True)
    data.dropna(subset=['token_no_stop'], inplace=True)

    data.reset_index(drop=True, inplace=True)

    return data



# --- FEATURE ENGINGEERING ----------------------------------------------------------------------------------

def feature_extraction(data):

    data = nlp_features(data)
    data = nlp_features_2(data)

    data

    data = regex_features(data)

    return data


def nlp_features(data):

    spacy_pipe = nlp.pipe(data["token_lower"].values, disable=["ner", "parser"])

    # get lemma, pos, previous lemma, previous pos, next lemma, next pos 
    features_gen = ((doc[0].lemma_, doc[0].pos_) for doc in spacy_pipe)
    data["lemma"], data["pos"] = zip(*features_gen)
    data["prev_lemma"] = data["lemma"].shift(periods=1)
    data["next_lemma"] = data["lemma"].shift(periods=-1)
    data["prev_pos"] = data["pos"].shift(periods=1)
    data["next_pos"] = data["pos"].shift(periods=-1)

    # trying some things out with stemmers 
    snow = SnowballStemmer(language='english')
    data["snowball_stemmer"] = data.apply(lambda row: snow.stem(row["token_lower"]), axis=1)
    port = PorterStemmer()
    data['Porter_stemmer'] = data.apply(lambda row: port.stem(row["token_lower"]), axis=1)

    return data


# Function to get the wordnet POS, it fixes compatibility issues with the nltk POS
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def nlp_features_2(data):

    # How to set up depencency parser: https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK
    # Command line instruction to start server
        #java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
        # -preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
        # -status_port 9000 -port 9000 -timeout 15000 & 

    dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
    lemmatizer = WordNetLemmatizer()

    column_values = data[['annotator']].values.ravel()
    annotator_ids = pd.unique(column_values)

    pos_tags = []
    heads = []
    dep_rels = []
    lemmas = []

    for annotator in annotator_ids:
        annotator_data = data[data['annotator'] == annotator]
        column_values = annotator_data[['sentence_id']].values.ravel()
        sentence_ids = pd.unique(column_values)

        for sent_id in sentence_ids:
            sentence = annotator_data.loc[annotator_data['sentence_id'] == sent_id, 'token']
            parse, = dep_parser.parse(sentence)
            conll = parse.to_conll(4) # get the conll format
            df = pd.DataFrame([x.split('\t') for x in conll.split('\n')[:-1]], columns=['word', 'pos', 'head', 'deprel'])
            df['head'] = df['head'].astype(int)
            head = list(df['head'].values)
            dep_rel = list(df['deprel'].values)

            for p, h, d in zip(nltk.pos_tag(sentence), head, dep_rel):
                pos_tags.append(p[1])
                heads.append(h)
                dep_rels.append(d)
                if get_wordnet_pos(p[1]): 
                    lemma = lemmatizer.lemmatize(p[0], pos=get_wordnet_pos(p[1]))
                else: 
                    lemma = lemmatizer.lemmatize(p[0])
                lemmas.append(lemma)
                    
                            
    data['pos_tag'] = pos_tags
    data['head'] = heads
    data['dependency'] = dep_rels
    data['lemma'] = lemmas
    data['is_part_of_negation'] = 0
    
    return data


def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))

    return results


def part_of_negation(data):

    multi_neg_list = ['by no means', 'on the contrary', 'not for the world', 'nothing at all', 'rather than', 'no more', 'no longer']
    data.token = data.token.str.lower()
    tokens = list(data.token.values)

    for exp in multi_neg_list:
        exp = exp.split(' ')
        index = find_sub_list(exp, tokens)
        for i in index:
            data.loc[data.index[i[0]], 'is_part_of_negation'] = 1
            data.loc[data.index[i[1]], 'is_part_of_negation'] = 1
            if i[1] - i[0] >1:
                data.loc[data.index[i[1]-1], 'is_part_of_negation'] = 1

    return data


def regex_features(data):

    prefixes = ['in', 'un', 'non', 'de', 'dis', 'a', 'anti', 'im', 'il', 'ir']
    postfixes = ('less') # also use for infix

    has_prefix = []
    has_postfix = []
    has_infix = []
    has_apostrophe = []
    base_tokens = []
    bases_in_dict = []

    multi_word = [] # by no means, not for the world, on the contrary, nothing at all, rather than


    for index, row in enumerate(data.values): 

        token = row[5] # matches 'token_lower' column
        
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
        
    data['has_prefix'] = has_prefix
    data['has_postfix'] = has_postfix
    data['has_infix'] = has_infix
    data['base'] = base_tokens
    data['base_in_dictionary'] = bases_in_dict
    data['has_apostrophe'] = has_apostrophe

    return data


# --- MAIN --------------------------------------------------------------------------------------------------

def main():

    data = pd.read_csv(DATA_DIR + 'SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt', sep="\t", header=None)
    data.columns = ['annotator', 'sentence_id', 'token_id', 'token', 'label']
    # exploration(data)
    # get_statistics_ravi(data)
    preprocessed_data = preprocessing(data)
    featurized_data = feature_extraction(preprocessed_data)
    data.to_csv('SEM2012_training_data_with_features.csv', index=False)

    # print('\nPREPROCESSED DATAFRAME\n')
    # print(preprocessed_data)
    # print('\nFEATURIZED DATAFRAME\n')
    # print(featurized_data)
    # print(featurized_data.columns)

    validation_data = pd.read_csv(DATA_DIR + 'SEM-2012-SharedTask-CD-SCO-dev-simple.v2.txt', sep="\t", header=None)
    validation_data.columns = ['annotator', 'sentence_id', 'token_id', 'token', 'label']
    # get_statistics_ravi(validation_data)
    validation_data = preprocessing(validation_data)
    validation_data = feature_extraction(validation_data)
    validation_data.to_csv('SEM2012_validation_data_with_features.csv', index=False)


if __name__ == '__main__':
    main()
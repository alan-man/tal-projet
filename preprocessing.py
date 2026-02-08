import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter

from nltk.stem import WordNetLemmatizer as wnl, SnowballStemmer as ss # for different language
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

import re, os, spacy, string, unicodedata, codecs    # for french lemming spacy

from typing import Literal

# USE SPARSE MATRIX !!

# different things to test
# TODO
# List different pre processing techniques

# A) Transformation paramétrique du texte (pre-traitements)

# Vous devez tester, par exemple, les cas suivants:
# - transformation en minuscule ou pas
# - suppression de la ponctuation
# - transformation des mots entièrement en majuscule en marqueurs spécifiques
# - suppression des chiffres ou pas
# - conservation d'une partie du texte seulement (seulement la première ligne = titre, seulement la dernière ligne = résumé, ...)
# - stemming
# - ...

# Vérifier systématiquement sur un exemple ou deux le bon fonctionnement des méthodes sur deux documents (au moins un de chaque classe).


# -------------- dataset

def load_pres(fname):
    """
    Charge les données
    """
    # M = -1, C = 1
    alltxts = []
    alllabs = []
    s=codecs.open(fname, 'r','utf-8') # pour régler le codage
    while True:
        txt = s.readline()
        if(len(txt))<5:
            break
        #
        lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*","\\1",txt)
        txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",txt)
        if lab.count('M') >0:
            alllabs.append(-1)
        else:
            alllabs.append(1)
        alltxts.append(txt)

    return alltxts,alllabs

def load_movies(path2data): # 1 classe par répertoire
    alltxts = [] # init vide
    labs = []
    cpt = 0
    for cl in os.listdir(path2data): # parcours des fichiers d'un répertoire
        for f in os.listdir(path2data+cl):
            txt = open(path2data+cl+'/'+f).read()
            alltxts.append(txt)
            labs.append(cpt)
        cpt+=1 # chg répertoire = cht classe

    return alltxts,labs

# -------------- preprocessing

def lemma_stem(text:str, 
               norm : Literal["lemma", "stem"] = "lemma", 
               lang : Literal["english", "french"] = "english",
               cap_name: bool = True, # keep names in Capital
               lem = None, stem = None): #  to avoid redefining each time
    
    """ Lemmatize/Stemm text. Input and output are text. No stemming or lemma if all caps """
    if not lem:
        lem = spacy.load("en_core_web_sm") if lang == "english" else spacy.load("fr_core_news_sm") # pasisng in argument to avoid redefining
    
    doc_lemmed = lem(text)

    if norm == 'lemma':
        if cap_name:
            tokens = [token.lemma_ if token.pos_ != 'PROPN' and not token.text.isupper() else token.text for token in doc_lemmed]
        else:
            tokens = [token.lemma_ for token in doc_lemmed]
    else:
        if not stem:
            stem = ss(lang) 
        tokens = [stem.stem(token.text) if token.pos_ != 'PROPN' and not token.text.isupper() else token.text for i, token in enumerate(doc_lemmed)]
    
    return " ".join(tokens)

def lower_case(text : str, all_cap : bool = False):
    """ Lower case text and if keep all caps words. Don't recognize names to keep them in cap."""

    if all_cap:
        return re.sub(
            r'\b\w+\b',
            lambda m: m.group(0) if m.group(0).isupper() else m.group(0).lower(), # case "HELLO," not include ","
            text
        )

    return text.lower()

def bow_stop_words(text: str, lang : Literal["english", "french"] = "english"): # mix of both?

    lst_stop_w = stopwords.words('english') if lang == "english" else stopwords.words('french')

    vectorizer = CountVectorizer(stop_words=lst_stop_w)

    X = vectorizer.fit_transform(text)
    return X, vectorizer

def preprocessing(text: str, 
                  low_case = True,
                  rm_punctuation = True,
                  rm_number = True,
                  word_norm : Literal["lemma", "stem"] = "lemma",
                  pos_tagging = False, # to check
                  all_capital = True, # keep all capital words as they are
                  cap_name = True, # garder les noms en majuscules ?
                  rm_accent = True, 
                  lang : Literal["english", "french"] = "english",
                  punct = string.punctuation + '\n\r\t', # punctuation can contain -, that would be kept
                  urls : bool = False) -> str: 
    
    """ Réalise le pré-processing du texte. Renvoie les tokens"""
    
    # conservation d'une partie du texte? 

    if word_norm: text = lemma_stem(text, word_norm, lang, cap_name)

    if pos_tagging: # used in keyword filtering/extraction 
        tokens = word_tokenize(text)
        pos_tags_words = pos_tag(tokens) # tuple (word, tag) -> check can filter on what

    if rm_accent: text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8") 
    if rm_punctuation: text = text.translate(str.maketrans(punct, ' ' * len(punct)))
    if low_case: text = lower_case(text, all_capital)
    if rm_number: text = re.sub('[0-9]+', '', text)
    if not urls : text = re.sub(r'https?://\S+|www\.\S+', 'URL', text) 

    if pos_tagging:
        return text, pos_tags_words
    return text

if __name__ == "__main__":
    text = "Alice and Bob are both, uncertain to go together. To SU university. héhéhe."
    print(text)
    text = lemma_stem(text, "lemma", "english")
    print("\nLemma\n",text)

    text = "Alice and Bob are both, uncertain to go together. To SU university. héhéhe."
    text = lemma_stem(text, "stem", "english")
    print("\nStem\n",text)

    text = "Alice and Bob are both, uncertain to go together. To SU university. héhéhe."
    text = preprocessing(text, low_case=False) # stem/lem already lowercase
    print("\nPreprocessing test \n" , text)
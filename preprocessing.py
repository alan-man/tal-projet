import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter

from nltk.stem import WordNetLemmatizer as wnl, SnowballStemmer  # for different language
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

# ------------- GLOBAL LOADING

NLP_EN = spacy.load("en_core_web_sm", disable=["parser", "ner"])
NLP_FR = spacy.load("fr_core_news_sm", disable=["parser", "ner"])


STEM_EN = SnowballStemmer("english")
STEM_FR = SnowballStemmer("french")

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
               pos_tag: bool = False):
    
    """ Lemmatize/Stemm text. Input and output are text. No stemming or lemma if all caps """
    
    lem = NLP_EN if lang == "english" else NLP_FR 
    stem = STEM_EN if lang == "english" else STEM_FR
    
    doc_lemmed = lem(text)

    tokens = []
    pos_tags = []

    for token in doc_lemmed:

        # Keep proper nouns or ALL CAPS
        if cap_name and (token.pos_ == "PROPN" or token.text.isupper()):
            word = token.text

        else:
            if norm == "lemma":
                word = token.lemma_
            else:  # stem
                word = stem.stem(token.text)

        tokens.append(word)

        if pos_tag:
            pos_tags.append((word, token.pos_))

    processed_txt = " ".join(tokens)

    if pos_tag:
        return processed_txt, pos_tags
    return processed_txt

def lower_case(text : str, all_cap : bool = False):
    """ Lower case text and if keep all caps words. Don't recognize names to keep them in cap."""

    if all_cap:
        return re.sub(
            r'\b\w+\b',
            lambda m: m.group(0) if m.group(0).isupper() else m.group(0).lower(), # case "HELLO," not include ","
            text
        )

    return text.lower()

# pos cannot be returned in count vectorizer
def preprocessing(text: str, 
                  low_case = True,
                  rm_punctuation = True,
                  rm_number = False,
                  word_norm : None|Literal["lemma", "stem"] = None, # stem faster
                  pos_tagging = False, # to check
                  all_capital = True, # keep all capital words as they are
                  cap_name = True, # garder les noms en majuscules 
                  rm_accent = True, 
                  lang : Literal["english", "french"] = "english",
                  punct = string.punctuation + '\n\r\t', # punctuation can contain -, that would be kept
                  urls : bool = True) -> str: # keep urls or not
    
    """ Réalise le pré-processing du texte. Renvoie les tokens"""
    # conservation d'une partie du texte? 

    if not urls : text = re.sub(r'https?://\S+|www\.\S+', 'URL', text) 

    if word_norm:
        if pos_tagging:
            text, pos_tags = lemma_stem(text, word_norm, lang, cap_name, pos_tagging)
        else:
            text = lemma_stem(text, word_norm, lang, cap_name, pos_tagging)

    if low_case: text = lower_case(text, all_capital)
    if rm_number: text = re.sub('[0-9]+', '', text)
    if rm_accent: text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8") 
    if rm_punctuation: text = text.translate(str.maketrans(punct, ' ' * len(punct)))

    if pos_tagging:
        return text, pos_tags
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
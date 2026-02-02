import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer as wnl, SnowballStemmer as ss # for different language
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import string
import unicodedata
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import re
import codecs
from typing import Literal

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


# -------------- preprocessing

def preprocessing(text: str, 
                  lower_case = True,
                  rm_punctuation = True,
                  rm_number = True,
                  lemming = True,
                  stemming = True,
                  pos_tagging = True,
                  all_capital = True, # check if keep
                  maj_name = True, # garder les noms en majuscules ?
                  rm_accent = True, # non normalized char check
                  language : Literal["english", "french"] = "english",
                  punct = string.punctuation) -> str: # punctuation can contain -, that would be kept
    
    """ Réalise le pré-processing du texte."""
    
    # conservation d'une partie du texte? 
    # check if langage frech or english

    if rm_accent: text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8") 

    if rm_punctuation: text = text.translate(str.maketrans(punct, ' ' * len(punct)))

    if lower_case:
        if all_capital:
            ' '.join(word if word.isupper() else word.lower() for word in text.split())
        else:
            text = text.lower()

    if rm_number: text = re.sub('[0-9]+', '', text)

    # working on tokens
    tokens = word_tokenize(text)

    if lemming:
        # for all word
        lemmer = wnl()
        for i, token in enumerate(tokens):
            tokens[i] = lemmer.lemmatize(token)

    if stemming:
        stemmer = ss(language)
        for i, token in enumerate(tokens):
            tokens[i] = stemmer.stem(token)

    if pos_tagging:
        pass

    # tags used for pos lemmatizing?

    # keeps names as they are with pos tagging

    text = " ".join(text)

    return text

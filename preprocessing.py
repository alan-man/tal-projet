import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer as wnl  
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import string
import unicodedata
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import re
import codecs

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
                  stemming = True,
                  pos_tagging = True,
                  maj_name = True, # ?
                  rm_accent = True, # non normalized char check
                  punct_lst = string.punctuation) -> str:
    """ Réalise le pré-processing du texte."""
    
    # conservation d'une partie du texte? 

    # check order!!
    if lower_case:
        text = text.lower()
    if rm_number:
        pass

    return 

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

class Preprocessing:
    """ 
    Class to realise the preprocessing.
    """

    def __init__(self, 
                low_case = True,
                rm_punctuation = True,
                rm_number = False,
                word_norm : None|Literal["lemma", "stem"] = None, # stem faster
                pos_tagging = False, # to check
                all_capital = True, # keep all capital words as they are
                cap_name = True, # garder les noms en majuscules, for pos tag
                rm_accent = True, 
                lang : Literal["english", "french"] = "english",
                punct = string.punctuation + '\n\r\t', # punctuation can contain -, that would be kept
                urls : bool = True 
                ):
        """Store preprocessing configuration and prepare language tools."""

        # basic flags
        self.low_case = low_case
        self.rm_punctuation = rm_punctuation
        self.rm_number = rm_number
        self.word_norm = word_norm
        self.pos_tagging = pos_tagging
        self.all_capital = all_capital
        self.cap_name = cap_name
        self.rm_accent = rm_accent
        self.lang = lang
        self.punct = punct
        self.urls = urls

        # choose language specific resources
        if lang == "english":
            self.nlp = NLP_EN
            self.stemmer = STEM_EN
            try:
                self.stopwords = set(stopwords.words("english"))
            except LookupError:
                # nltk data may not be downloaded yet
                import nltk
                nltk.download('stopwords')
                self.stopwords = set(stopwords.words("english"))
                
        else:
            self.nlp = NLP_FR
            self.stemmer = STEM_FR
            try:
                self.stopwords = set(stopwords.words("french"))
            except LookupError:
                import nltk
                nltk.download('stopwords')
                self.stopwords = set(stopwords.words("french"))

        # placeholder for any additional initializations
        # e.g. a regex for url removal
        self._url_pattern = re.compile(r'https?://\S+|www\.\S+')

    def __str__(self):
        # Filter only the attributes you want, or just print them all
        output = "Configuration Flags:\n"
        for key, value in vars(self).items():
            output += f"  {key}: {value}\n"
        return output

    # functions to process
    def lower_case(self, text : str):
        """ Lower case text and if keep all caps words. Don't recognize names to keep them in cap."""

        if self.all_capital:
            return re.sub(
                r'\b\w+\b',
                lambda m: m.group(0) if m.group(0).isupper() else m.group(0).lower(), # case "HELLO," not include ","
                text
            )

        return text.lower()

    def lemma_stem(self, text:str):
    
        """Lemmatize/stem `text` according to instance settings.
        Proper nouns or ALL-CAPS tokens are preserved when ``cap_name`` is True.
        """
        # use the resources initialized on the object rather than globals
        lem = self.nlp
        stem = self.stemmer
        
        doc_lemmed = lem(text)

        tokens = []
        pos_tags = []

        for token in doc_lemmed:

            # Keep proper nouns or ALL CAPS
            if self.cap_name and (token.pos_ == "PROPN" or token.text.isupper()):
                word = token.text

            else:
                if self.word_norm == "lemma":
                    word = token.lemma_
                else:  # stem
                    word = stem.stem(token.text)

            tokens.append(word)

            if self.pos_tagging:
                pos_tags.append((word, token.pos_))

        processed_txt = " ".join(tokens)

        if self.pos_tagging:
            return processed_txt, pos_tags
        return processed_txt

    def process(self, text : str):
        # preprocess a text
        """Réalise le pré-processing du texte. Renvoie les tokens."""
        # conservation d'une partie du texte? 

        if not self.urls:
            # use compiled pattern instead of boolean flag
            text = re.sub(self._url_pattern, 'URL', text)

        if self.word_norm:
            if self.pos_tagging:
                text, pos_tags = self.lemma_stem(text)
            else:
                text = self.lemma_stem(text)

        if self.low_case: text = self.lower_case(text)
        if self.rm_number: text = re.sub('[0-9]+', '', text)
        if self.rm_accent: text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8") 
        if self.rm_punctuation: text = text.translate(str.maketrans(self.punct, ' ' * len(self.punct)))

        if self.pos_tagging:
            return text, pos_tags
        return text

    def __call__(self, text): # for count vectorizer
        return self.process(text)

# pos cannot be returned in count vectorizer

if __name__ == "__main__":
    prep = Preprocessing(word_norm="lemma")

    text = "Alice and Bob are both, uncertain to go together. To SU university. héhéhe."
    print(text)
    text = prep.process(text)
    print("\nLemma\n",text)

    prep = Preprocessing(word_norm="stem")
    text = "Alice and Bob are both, uncertain to go together. To SU university. héhéhe."
    text = prep.process(text)
    print("\nStem\n",text)

    prep = Preprocessing()
    text = "Alice and Bob are both, uncertain to go together. To SU university. héhéhe."
    text = prep.process(text) # stem/lem already lowercase
    print("\nPreprocessing test \n" , text)
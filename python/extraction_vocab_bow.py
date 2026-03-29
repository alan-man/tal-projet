import numpy as np
import matplotlib.pyplot as plt
import codecs
import re
import unicodedata
import string
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from python.preprocessing_class import Preprocessing


def basic_bow_exploration(list_txts,lang="french"):

    punc = set(string.punctuation + '\n\r\t')
    if lang == "french":
        punc.remove("'") # keep the ' for french words
    custom_punctuation = "".join(punc)

    prep = Preprocessing(low_case = True,
        rm_punctuation = True,
        rm_number = False,
        word_norm = None, # stem faster
        pos_tagging = False, # to check
        all_capital = True, # keep all capital words as they are
        cap_name = True, # garder les noms en majuscules, for pos tag
        rm_accent = False, 
        lang = lang,
        punct = custom_punctuation, # punctuation can contain -, that would be kept
        urls = False 
    )

    cleaned_txts = [prep.process(text) for text in list_txts]

    #  basic bow
    vectorizer1 = CountVectorizer()
    X1 = vectorizer1.fit_transform(list_txts)
    vocab1 = vectorizer1.get_feature_names_out()

    print(f"{X1.shape[0]} documents")
    print(f"taille origine du vocab: {X1.shape[1]} mots total")
    print(vocab1,"\n")

    # 100 most frequent
    vectorizer2 = CountVectorizer(max_features=100)
    X2 = vectorizer2.fit_transform(list_txts)
    vocab2 = vectorizer2.get_feature_names_out()
    print(f"vocab avec les 100 mots plus frequents :")
    print(vocab2,"\n")

    # tf idf
    vect_tfidf = TfidfVectorizer(use_idf= True, smooth_idf=True, sublinear_tf=False)
    X3 = vect_tfidf.fit_transform(list_txts)
    vocab3 = vect_tfidf.get_feature_names_out()
    idf = vect_tfidf.idf_
    
    # doc freq = inverse of idf 
    # so getting the biggest df means getting the smallest idf 
    idx_top_df = np.argsort(idf)[:100]
    words_top_df = vocab3[idx_top_df] 

    print(f"100 mots avec la plus grande frequence doc:")
    print(words_top_df)

    # odds ratio TODO



    # zipf distribution 
    total_counts = np.asarray(X1.sum(axis=0)).ravel()
    sorted_counts = np.sort(total_counts)[::-1] # descending order
    ranks = np.arange(1,len(sorted_counts)+1)
    plt.plot(ranks,sorted_counts)
    plt.xlabel("rank")
    plt.ylabel("freq")
    plt.title("Zipf distribution")
    plt.show()

    plt.loglog(ranks,sorted_counts)
    plt.plot(ranks,sorted_counts)
    plt.xlabel("rank")
    plt.ylabel("freq")
    plt.title("Zipf distribution log scale")
    plt.show()

    # bigrams
    vectorizer_bi = CountVectorizer(ngram_range=(2, 2), max_features=100)
    X_bi = vectorizer_bi.fit_transform(list_txts)

    vocab_bi = np.array(vectorizer_bi.get_feature_names_out())
    print("\n100 bigrammes les plus freq:")
    print(vocab_bi)

    # trigrams
    vectorizer_tri = CountVectorizer(ngram_range=(3, 3), max_features=100)
    X_tri= vectorizer_tri.fit_transform(list_txts)

    vocab_tri = np.array(vectorizer_tri.get_feature_names_out())
    print("\n100 trigrammes les plus freq:")
    print(vocab_tri)

import numpy as np
import matplotlib.pyplot as plt
import codecs
import re
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from preprocessing import preprocessing


def exploration(list_txts,preprocess):
    #  basic bow
    vectorizer1 = CountVectorizer(preprocessor=preprocess)
    X1 = vectorizer1.fit_transform(list_txts)
    vocab1 = vectorizer1.get_feature_names_out()

    print(f"{X1.shape[0]} documents")
    print(f"taille origine du vocab: {X1.shape[1]} mots total")
    print(vocab1,"\n")

    # 100 most frequent
    vectorizer2 = CountVectorizer(preprocessor=preprocess,max_features=100)
    X2 = vectorizer2.fit_transform(list_txts)
    vocab2 = vectorizer2.get_feature_names_out()
    print(f"vocab avec les 100 mots plus frequents :")
    print(vocab2,"\n")

    # tf idf
    vect_tfidf = TfidfVectorizer(preprocessor=preprocess,use_idf= True, smooth_idf=True, sublinear_tf=False)
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
    vectorizer_bi = CountVectorizer(preprocessor=preprocess,ngram_range=(2, 2), max_features=100)
    X_bi = vectorizer_bi.fit_transform(list_txts)

    vocab_bi = np.array(vectorizer_bi.get_feature_names_out())
    print("\n100 bigrammes les plus freq:")
    print(vocab_bi)

    # trigrams
    vectorizer_tri = CountVectorizer(preprocessor=preprocess,ngram_range=(3, 3), max_features=100)
    X_tri= vectorizer_tri.fit_transform(list_txts)

    vocab_tri = np.array(vectorizer_tri.get_feature_names_out())
    print("\n100 trigrammes les plus freq:")
    print(vocab_tri)

# sur movies
data_text, data_label

import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

text = [t.split() for t in data_text] # p is the class

# the following configuration is the default configuration
w2v = gensim.models.word2vec.Word2Vec(sentences=text,
                                vector_size=100, window=5,               ### here we train a cbow model
                                min_count=5,
                                sample=0.001, workers=3,
                                sg=1, hs=0, negative=5,        ### set sg to 1 to train a sg model
                                cbow_mean=1, epochs=5)
# sauvegarder modèle !!!!!
w2v.save("W2v-movies.dat")
# pour load
# w2v = gensim.models.Word2Vec.load("W2v-movies.dat")


def vectorize(text, w2v, mean=False, min = False, max = False):
    """
    This function should vectorize one review

    input: str
    output: np.array(float)
    """ 
    vec = []
    keys = w2v.wv.key_to_index.keys()

    # sum
    for word in text:
        if word in keys:
            idx = w2v.wv.key_to_index[word] 
            vec.append(w2v.wv[idx])
            
    if mean:
        return np.mean(vec, axis = 0)
    if min:
        return np.min(vec, axis = 0)
    if max:
        return np.max(vec, axis = 0)
    return np.sum(vec, axis = 0)

from sklearn import preprocessing

X = [vectorize(text, w2v) for text in X_train]
Xtest = [vectorize(text, w2v) for text in X_test]

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
Xtest_scaled = scaler.transform(Xtest)
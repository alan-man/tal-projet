from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
import numpy as np
from nltk.corpus import stopwords
import json, time, re, codecs, string
from preprocessing_class import Preprocessing, load_pres

# utilities

punc = set(string.punctuation + '\n\r\t')
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
                lang = "french",
                punct = custom_punctuation, # punctuation can contain -, that would be kept
                urls = False 
                )

alltxt, alllabs = load_pres("Dataset/corpus.tache1.learn.utf8")
alltxt, alllabs = np.array(alltxt), np.array(alllabs)

print("data loaded")

# train test split here for test in final prediction

X_train, X_test, y_train, y_test = train_test_split(
    alltxt, alllabs, 
    test_size=0.20, 
    stratify=alllabs, 
    random_state=42
)

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, 
    test_size=0.20, 
    stratify=y_train, 
    random_state=42
)


final_stopwords_list = stopwords.words('french') # stopwords.words('english') 

def stopwords_rmv(text, stopwords=final_stopwords_list):
    return " ".join([word for word in text.split() if word not in stopwords])

# loading data
all_texts = [stopwords_rmv(prep.process(t)) for t in X_train]
all_texts_tok = [t.split() for t in all_texts]

RESULTS = []

def log_result(name, metrics):
    RESULTS.append({"model": name, **metrics})
    print(f"{name:40s} : F1={metrics['F1']:.4f}  AUC={metrics['AUC']:.4f}")

def mean_embed_wv(texts, model):
    """Word2Vec / FastText: mean pool over known words."""
    vecs = []
    for text in texts:
        words = text.split()
        ws = [model.wv[w] for w in words if w in model.wv]
        vecs.append(np.mean(ws, axis=0) if ws else np.zeros(model.vector_size))
    return np.array(vecs)

def run_cv(X, y, n_splits=5, scaling = True):
    """Scale + LR inside a pipeline, return mean CV metrics."""
    if scaling :
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(solver="lbfgs", max_iter=2000))
        ])
    else:
        pipe = Pipeline([
            ("clf",    LogisticRegression(solver="lbfgs", max_iter=2000))
        ])
    print("running cv")
    scores = cross_validate(pipe, X, y, cv=n_splits,
                            scoring=["f1_macro", "roc_auc",
                                     "average_precision",
                                     "precision_macro", "recall_macro"])
    return {
        "F1":            scores["test_f1_macro"].mean(),
        "F1_std":        scores["test_f1_macro"].std(),
        "AUC":           scores["test_roc_auc"].mean(),
        "AUC_std":       scores["test_roc_auc"].std(),
        "AvgPrec":       scores["test_average_precision"].mean(),
        "AvgPrec_std":   scores["test_average_precision"].std(),
        "Precision":     scores["test_precision_macro"].mean(),
        "Precision_std": scores["test_precision_macro"].std(),
        "Recall":        scores["test_recall_macro"].mean(),
        "Recall_std":    scores["test_recall_macro"].std(),
    }


# cleaner helper that works directly with KeyedVectors
def mean_embed_kv(texts, kv):
    vecs = []
    for text in texts:
        words = text.split()
        ws = [kv[w] for w in words if w in kv]
        vecs.append(np.mean(ws, axis=0) if ws else np.zeros(kv.vector_size))
    return np.array(vecs)

kv_vec = KeyedVectors.load_word2vec_format("cc.fr.300.vec", binary=False)
log_result("cc.fr.300.vec (W2V-like)", run_cv(mean_embed_kv(all_texts, kv_vec), y_train))

ft_bin = load_facebook_model("cc.fr.300.bin")
log_result("cc.fr.300.bin (FastText)",  run_cv(mean_embed_wv(all_texts, ft_bin), y_train))

# saving
with open("w2v_ft_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

print("saved")
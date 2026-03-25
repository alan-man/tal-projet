from preprocessing_class import Preprocessing, load_pres
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from nltk.corpus import stopwords
import numpy as np
import string
import matplotlib.pyplot as plt
import pandas as pd

# load dataset
alltxt, alllabs = load_pres("Dataset/corpus.tache1.learn.utf8")
alltxt, alllabs = np.array(alltxt), np.array(alllabs)
# M = -1, C = 1

alllabs = np.where(alllabs == 1, 0, 1)

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

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier

final_stopwords_list = stopwords.words('french') # stopwords.words('english') 

def build_vectorizer(name="count", stop_words=None, max_features=None,
                     lowercase=False, strip_accents=None, ngram_range=(1,1), use_idf= True, smooth_idf=True, sublinear_tf=False, 
                     max_df=0.95, min_df=2, preprocessor = lambda x: x):
    if name == "count":
        vectorizer = CountVectorizer(stop_words=stop_words,max_features=max_features,max_df=max_df,min_df=min_df,ngram_range=ngram_range,lowercase=lowercase,
                                     strip_accents=strip_accents, preprocessor=preprocessor)    
        
    elif name == "tfidf":
        vectorizer = TfidfVectorizer(use_idf= use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf,max_df=max_df, min_df=min_df, max_features=max_features,
                                     stop_words=stop_words, ngram_range=ngram_range,lowercase=lowercase,strip_accents=strip_accents, preprocessor=preprocessor)
    
    return vectorizer

def vectorize_txts(txts,vectorizer):
    X = vectorizer.fit_transform(txts)
    return X

## solver = 'newton-cholesky' to try for logistic regression (n_samples >> n_features * n_classes, especially with one-hot encoded categorical features with rare categories)
def build_model(name="kmeans", n_clusters = 2, max_iter = 1000, penalty='l2', loss='squared_hinge',kernel='rbf',solver='lbfgs'):
    if name == "kmeans":
        return KMeans(n_clusters=n_clusters,max_iter=max_iter) 
    elif name == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    elif name == "linear_svm":
        return LinearSVC(class_weight="balanced", penalty=penalty,loss=loss,max_iter=max_iter)
    elif name == "nb":
        return MultinomialNB()
    elif name == "svm":
        return SVC(max_iter=max_iter,kernel=kernel)
    elif name == "logreg":
        return LogisticRegression(class_weight="balanced", solver=solver,max_iter=max_iter) 
    
nb = build_model(name="nb")

vect_count = build_vectorizer(name="count", stop_words=final_stopwords_list, preprocessor=prep)
vect_tf = build_vectorizer(name="tfidf", stop_words=final_stopwords_list, preprocessor=prep)

# with simple grid search
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
import numpy as np

# --- Pipeline: vectorizer + model ---
pipe = Pipeline([
    ('tfidf', vect_tf),
    ('clf', nb)
])

# --- Tune both vectorizer and model params ---
param_grid = {
    'clf__alpha': [0.1, 1],
}

scoring = {
    'f1': 'f1_macro',
    'precision': 'precision_macro',
    'recall': 'recall_macro',
    'roc_auc': 'roc_auc_ovr',        # one-vs-rest for multiclass
    'avg_precision': 'average_precision_macro'
}

# --- Grid search with cross-validation on train set only ---
grid_search = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    refit='f1', 
    scoring=scoring,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# --- CV results ---
print("Best params:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)

# --- Evaluate on test set ---
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
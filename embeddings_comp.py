from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
import numpy as np
import codecs, re, gc, time, json
from sklearn.pipeline import Pipeline

from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score

def run_cv(X, y, n_splits=5):
    """Scale + LR inside a pipeline, return mean CV metrics."""
    pipe = Pipeline([
        # ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(solver="lbfgs", max_iter=2000))
    ])
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

# load dataset
alltxt, alllabs = load_pres("Dataset/corpus.tache1.learn.utf8")
alltxt, alllabs = np.array(alltxt), np.array(alllabs)

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

print("laoded dataset")

from sentence_transformers import SentenceTransformer

# Load model (allow custom code)
# model = SentenceTransformer(
#     "intfloat/multilingual-e5-base"
# )
model = SentenceTransformer("./model_emb")
print("model loaded")

# running

dims = [128, 256, 512, 768, 1024]
rows = []

# encode once at full size
t0 = time.perf_counter()
X_train_full = model.encode(
    X_train, batch_size=32, show_progress_bar=True,
    task="classification"
)
X_test_full = model.encode(
    X_test, batch_size=32, show_progress_bar=True,
    task="classification"
)
t1 = time.perf_counter()
full_encode_s = t1 - t0

for d in dims:
    print("Running dim ", d)
    # slice instead of re-encoding
    X_train_emb = X_train_full[:, :d]
    X_test_emb = X_test_full[:, :d]

    t3 = time.perf_counter()
    metrics = run_cv(X_train_emb, y_train)
    t4 = time.perf_counter()

    # test evaluation

    clf = LogisticRegression(solver="lbfgs", max_iter=2000)
    clf.fit(X_train_emb, y_train)

    y_pred = clf.predict(X_test_emb)
    ypred_prob = clf.predict_proba(X_test_emb)[:, 1]

    test_metrics = {
        "f1": f1_score(y_test, y_pred, average="macro"),
        "auc": roc_auc_score(y_test, ypred_prob),
        "avg_prec": average_precision_score(y_test, ypred_prob),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
    }
    
    rows.append({
        "dim":            d,
        "encode_total_s": full_encode_s,  # same for all, encoding was done once
        "eval_s":         t4 - t3,
        **metrics,
        **test_metrics
    })

with open("embed_size_cv.json", "w") as f:
    json.dump(rows, f, indent=2)

del X_train_full, X_test_full, X_test_emb, X_train_emb
gc.collect()

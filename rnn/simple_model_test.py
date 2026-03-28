# Simple model tester with cross-validation: Test different models with various preprocessing and vectorization options.
# Uses 5-fold cross-validation on train set

import json
import numpy as np
import pandas as pd
import string
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from nltk.corpus import stopwords

from preprocessing_class import Preprocessing, load_movies, load_pres

# config
DATASET = "pres"
RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_FEATURE = 5000
SAVING_FILE_NAME =  DATASET + "_simple_model_2"


STOPWORDS_FR = stopwords.words('french')
STOPWORDS_EN = stopwords.words('english')

STOPWORDS = STOPWORDS_FR if DATASET == "pres" else STOPWORDS_EN

PUNC = set(string.punctuation + '\n\r\t')
PUNC.discard("'")  # keep apostrophes for French words
CUSTOM_PUNCTUATION = "".join(PUNC)

def get_preprocessing_configs():
    """Returns a dict of preprocessing configurations to test."""

    lang = "french" if DATASET == "pres" else "english"
    
    configs = {
        "raw_text": None,
        "no_lower": Preprocessing(
            low_case=False,
            rm_punctuation=True,
            rm_number=False,
            word_norm=None,
            pos_tagging=False,
            all_capital=True,
            cap_name=True,
            rm_accent=False,
            lang=lang,
            punct=CUSTOM_PUNCTUATION,
            urls=False
        ),
    }
    return configs

def get_vectorizer_configs():

    """Returns a dict of vectorizer configurations to test."""
    configs = {
        "count_unigram": CountVectorizer(
            stop_words=STOPWORDS,
            max_features=MAX_FEATURE,
            max_df=0.95,
            min_df=2,
            ngram_range=(1, 1),
            lowercase=False,
            preprocessor= lambda x: x
        ),
        "count_bigram": CountVectorizer(
            stop_words=STOPWORDS,
            max_features=MAX_FEATURE,
            max_df=0.95,
            min_df=2,
            ngram_range=(1, 2),
            lowercase=False,
            preprocessor= lambda x: x
        ),
        "tfidf_unigram": TfidfVectorizer(
            stop_words=STOPWORDS,
            max_features=MAX_FEATURE,
            max_df=0.95,
            min_df=2,
            ngram_range=(1, 1),
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
            lowercase=False,
            preprocessor= lambda x: x
        ),
        "tfidf_bigram": TfidfVectorizer(
            stop_words=STOPWORDS,
            max_features=MAX_FEATURE,
            max_df=0.95,
            min_df=2,
            ngram_range=(1, 2),
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
            lowercase=False,
            preprocessor= lambda x: x
        ),
    }
    return configs

def get_model_configs():
    """Returns a dict of model configurations to test."""
    
    configs = {
        "naive_bayes": MultinomialNB(alpha=1.0),
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            solver='lbfgs',
            max_iter=5000,
            random_state=RANDOM_STATE
        ),
        "linear_svm": LinearSVC(
            class_weight="balanced",
            penalty='l2',
            loss='squared_hinge',
            max_iter=5000,
            random_state=RANDOM_STATE
        ),
        "svm_rbf": SVC(
            kernel='rbf',
            max_iter=5000,
            random_state=RANDOM_STATE,
            probability=True
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1
        ),
    }
    return configs

def compute_metrics(y_true, y_pred, y_proba=None):
    if DATASET == "pres":
        metrics = {
            "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average='weighted', zero_division=0),
            "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
        }
    else:
        metrics = {
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average='weighted', zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }

    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            metrics["avg_precision"] = average_precision_score(y_true, y_proba)
        except Exception as e:
            print(f"Warning: Could not compute AUC/AP: {e}")
            metrics["roc_auc"] = None
            metrics["avg_precision"] = None
    
    return metrics

def load_and_split_data(data_path, test_size=TEST_SIZE):
    print(data_path)
    print("DATASET ", DATASET)

    if DATASET == "pres":
        # Load and convert labels
        alltxt, alllabs = load_pres(data_path)
        alltxt = np.array(alltxt)
        # Convert labels: 1 -> 0, -1 -> 1
        alllabs = np.where(np.array(alllabs) == 1, 0, 1)
        
        print(f"Size train: {len(alltxt)}")
    else:
        alltxt,alllabs = load_movies(data_path)
        alltxt, alllabs = np.array(alltxt), np.array(alllabs)
        print(f"Size train: {len(alltxt)}")
        
    X_train, X_test, y_train, y_test = train_test_split(
        alltxt, alllabs,
        test_size=test_size,
        stratify=alllabs,
        random_state=RANDOM_STATE
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ==================== Experiment Runner ====================

def run_experiment(X_train, X_test, y_train, y_test,
                   prep_name, vectorizer_name, model_name,
                   preprocessor, vectorizer, model):
    
    print(prep_name, vectorizer_name, model_name)
    try:            
        pipe = Pipeline([
            ('vectorizer', vectorizer),
            ('scaler', StandardScaler(with_mean=False)),
            ('model', model)])
        
        if prep_name == "raw_text":
            X_train_prep = X_train
            X_test_prep = X_test
        else:
            X_train_prep = [preprocessor.process(str(t)) for t in X_train]
            X_test_prep = [preprocessor.process(str(t)) for t in X_test]
        
        if DATASET == "pres":
            cv_scoring = {
                'f1_macro': 'f1_macro',
                'precision_macro': 'precision_macro',
                'recall_macro': 'recall_macro',
            }
        else:
            cv_scoring = {
                'f1_macro': 'f1_macro',
                'precision_macro': 'precision_macro',
                'recall_macro': 'recall_macro',
                "accuracy" : "accuracy",
            }
            
        cv_results = cross_validate(
            pipe, X_train_prep, y_train,
            cv=5,
            scoring=cv_scoring,
            return_train_score=False
        )
        
        cv_f1_scores = cv_results['test_f1_macro']
        cv_proba_scores = []
        
        try:
            y_cv_proba = cross_val_predict(pipe, X_train_prep, y_train, cv=5, method='predict_proba')[:, 1] 
        except (AttributeError, ValueError):
            # decision_function
            try:
                y_cv_proba = cross_val_predict(
                    pipe, X_train_prep, y_train,
                    cv=5,
                    method='decision_function'
                )
                # Normalize to [0, 1]
                y_cv_proba = 1 / (1 + np.exp(-y_cv_proba))
            except (AttributeError, ValueError):
                y_cv_proba = None

        if DATASET == "pres":
            cv_metrics = {
                "f1_macro": np.mean(cv_results['test_f1_macro']),
                "f1_std": np.std(cv_results['test_f1_macro']),
                "precision_macro": np.mean(cv_results['test_precision_macro']),
                "recall_macro": np.mean(cv_results['test_recall_macro']),
                "roc_auc": None,
                "avg_precision": None,
            }
        else:
            cv_metrics = {
                "f1_macro": np.mean(cv_results['test_f1_macro']),
                "f1_std": np.std(cv_results['test_f1_macro']),
                "precision_macro": np.mean(cv_results['test_precision_macro']),
                "recall_macro": np.mean(cv_results['test_recall_macro']),
                "accuracy": np.mean(cv_results['test_accuracy']), # check of work
                "roc_auc": None,
                "avg_precision": None,
            }
        
        if y_cv_proba is not None:
            try:
                cv_metrics["roc_auc"] = roc_auc_score(y_train, y_cv_proba)
                cv_metrics["avg_precision"] = average_precision_score(y_train, y_cv_proba)
            except Exception as e:
                print(f"    Warning: Could not compute CV AUC/AP: {e}")
        
        pipe.fit(X_train_prep, y_train)
        y_test_pred = pipe.predict(X_test_prep)
        y_test_proba = None
        try:
            y_test_proba = pipe.predict_proba(X_test_prep)[:, 1]
        except (AttributeError, ValueError):
            try:
                scores = pipe.decision_function(X_test_prep)
                y_test_proba = 1 / (1 + np.exp(-scores))
            except (AttributeError, ValueError):
                pass
        
        test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)
        
        result = {
            "prep": prep_name,
            "vectorizer": vectorizer_name,
            "model": model_name,
            "cv_metrics": cv_metrics,
            "test_metrics": test_metrics,
            "status": "success"
        }
        
        print(f"(cv_f1: {cv_metrics['f1_macro']:.4f}, test_f1: {test_metrics['f1_macro']:.4f})")
        return result
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "prep": prep_name,
            "vectorizer": vectorizer_name,
            "model": model_name,
            "status": "error",
            "error_msg": str(e)
        }


def run_all_experiments(X_train, X_test, y_train, y_test):
    """Run all experiment combinations."""
    
    prep_configs = get_preprocessing_configs()
    vect_configs = get_vectorizer_configs()
    model_configs = get_model_configs()
    
    results = []
    total = len(prep_configs) * len(vect_configs) * len(model_configs)

    print("number of congfif total",total )    
    for prep_name, preprocessor in prep_configs.items():
        print(f"\nPreprocessing: {prep_name}")
        for vect_name, vectorizer in vect_configs.items():
            print(f"Vectorizer: {vect_name}")
            for model_name, model in model_configs.items():
                
                result = run_experiment(
                    X_train, X_test, y_train, y_test,
                    prep_name, vect_name, model_name,
                    preprocessor, vectorizer, model)
                results.append(result)
    
    return results

def save_results(results, output_file=f"{SAVING_FILE_NAME}.json"):
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    return output_file

if __name__ == "__main__":    
    path =  "Dataset/corpus.tache1.learn.utf8" if DATASET == "pres" else "./Dataset/movies1000/"
    X_train, X_test, y_train, y_test = load_and_split_data(path)
    results = run_all_experiments(X_train, X_test, y_train, y_test)

    output_file = f"{SAVING_FILE_NAME}{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_results(results, output_file)
    print("finished")
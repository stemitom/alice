import os
import pprint
import pickle
import scipy
import numpy as np
import pandas as pd
import warnings as ws
ws.filterwarnings('ignore')

#Sklearn functions
from contextlib import contextmanager
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression

#Catboost
from catboost import CatBoostClassifier

#Skopt functions
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, VerboseCallback, DeltaXStopper
from skopt.space import Real, Categorical, Integer
from time import time, ctime

DATA_DIR = '~/Desktop/python/DSML/ds_workspace/kaggle/datasets/alice/'
OUTPUT_DIR = '~/Desktop/python/DSML/ds_workspace/kaggle/codebase/alice/output/'
SEED = 17
N_JOBS = -1
NUM_TIME_SPLITS = 10
SITE_NGRAMS = (1,5)
MAX_FEATURES = 50000
BEST_LOGIT_C = 3.359818286283781
vectorizer_params = {'ngram_range': (1,5), 'max_features': 50000, 'tokenizer': lambda s: s.split}

times = ['time%s' % i for i in range(1,11)]
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_sessions.csv'), index_col='session_id', parse_dates=times)
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_sessions.csv'), index_col='session_id', parse_dates=times)
train_df.sort_values(by='time1', inplace=True)

@contextmanager
def timer(name):
    t = time()
    yield
    print(f"[{name}] done in {time() - t:.1f}s")

class DataPreparator(BaseEstimator, TransformerMixin):
    """
    Fill Nan with zero values
    """
    def fit(self, values):
        return self
    def transform(self, X, y=None):
        sites = ['site%s' % i for i in range(1,11)]
        return X[sites].fillna(0).astype('int')
    
class ListPreparator(BaseEstimator, TransformerMixin):
    """
    Prepare a CVectorizer 2-D from data
    """
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.values.tolist()
        return [" ".join([str(site) for site in row]) for row in X]

class AttributesAdder(BaseEstimator, TransformerMixin):
    """
    Add new attributes to training and test set.
    """
    def fit(self, X, y=None):
        return self 
    def transform(self, X, y=None):
        hour = X['time1'].apply(lambda ts: ts.hour)
        morning = ((hour >= 7) & (hour <= 11)).astype('int')
        day = ((hour >= 12) & (hour <= 18)).astype('int')
        evening = ((hour >= 19) & (hour <= 23)).astype('int')
        # night = ((hour >= 0) > (hour <= 6)).astype('int')
    
        
        month = X['time1'].apply(lambda ts: ts.month)
        summer = ((month >= 6) & (month <= 8)).astype('int')
        
        weekday = X['time1'].apply(lambda ts: ts.weekday()).astype('int')
        
        year = X['time1'].apply(lambda ts: ts.year).astype('int')

        X = np.c_[morning.values, day.values, evening.values, summer.values, weekday.values, year.values]
        return X

class ScaledAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Add attributes that needs to be scaled
    """
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        times = ['time%s' % i for i in range(1,11)]
        session_duration = (X[times].max(axis=1) - X[times].min(axis=1)).astype('timedelta64[s]').astype('int') ** .2
        number_of_sites = X[times].isnull().sum(axis=1).apply(lambda x: 10 - x)
        time_per_site = (session_duration / number_of_sites) ** .2
        # X = np.c_[session_duration.values, time_per_site.values]
        X = np.c_[session_duration.values]
        
        return X

def report_performance(optimizer, X, y, title, callbacks=None):
    start = time()
    if callbacks:
        optimizer.fit(X, y, callback=callbacks)
    else:
        optimizer.fit(X, y)
    d = pd.DataFrame(optimizer.cv_results_)
    best_score_ = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_
    print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
           +u"\u00B1"+" %.3f") % (time() - start, 
                                  len(optimizer.cv_results_['params']),
                                  best_score,
                                  best_score_std))
    print('Best parameters:')
    pprint.pprint(best_params)
    print()
    return best_params

def tune_with_bayes(X_train, y_train):
    roc_auc = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
    time_split = TimeSeriesSplit(n_splits=10)

    cboost = CatBoostClassifier(thread_count=2, od_type='Iter', verbose=False)
    search_spaces = {'iterations': Integer(10, 1000),
                    'depth': Integer(1, 8),
                    'learning_rate': Real(0.01, 1.0, 'log-uniform'),
                    'random_strength': Real(1e-9, 10, 'log-uniform'),
                    'bagging_temperature': Real(0.0, 1.0),
                    'border_count': Integer(1, 255),
                    'l2_leaf_reg': Integer(2, 30),
                    'scale_pos_weight':Real(0.01, 1.0, 'uniform')}

    opt = BayesSearchCV(cboost,
                        search_spaces,
                        scoring=roc_auc,
                        cv=time_split,
                        n_iter=100,
                        n_jobs=1,
                        return_train_score=False,
                        refit=True,
                        optimizer_kwargs={'base_estimator': 'GP'},
                        random_state=17)

    best_params = report_performance(opt, X_train, y_train,'CatBoost', 
                            callbacks=[VerboseCallback(100), 
                                        DeadlineStopper(60*10)])

    best_params['iterations'] = 1000
    tuned_model = CatBoostClassifier(**best_params, od_type='Iter',one_hot_max_size=10)
    # tuned_model = CatBoostClassifier(**best_params,task_type = "GPU",od_type='Iter',one_hot_max_size=10)
    tuned_model.fit(X_train,y_train)
    return tuned_model

def write_to_submission_file(predicted_labels, out_file, target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,index = np.arange(1, predicted_labels.shape[0] + 1), columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)

vectorizer_pipeline = Pipeline([
    ('preparator', DataPreparator()),
    ('list_preparator', ListPreparator()),
    # ('vectorizer', TfidfVectorizer(**vectorizer_params))
    ('vectorizer', CountVectorizer(ngram_range=(1,3), max_features=50000))
    # ('vectorizer', TfidfVectorizer(ngram_range=(1,3), max_features=50000, tokenizer=lambda s: s.split()))
])

attributes_pipeline = Pipeline([
    ('adder', AttributesAdder())
])

scaled_attribs_pipeline = Pipeline([
    ('adder', ScaledAttributesAdder()),
    ('scaler', StandardScaler())
])


full_pipeline = FeatureUnion(transformer_list=[
    ('vectorizer_pipeline', vectorizer_pipeline),
    ('attributes_pipeline', attributes_pipeline),
    ('scaled_attribs_pipeline', scaled_attribs_pipeline)
])

with timer("Processing dataset"):
    X_train = full_pipeline.fit_transform(train_df)
    X_test = full_pipeline.transform(test_df)
    y_train = train_df['target'].astype('int').values

with timer("Performing Logistic Regression Operations"):
    time_split = TimeSeriesSplit(n_splits=10)
    #The used C values has been computed by using GridSearchCV on np.logspace(-2,2,20)
    logit = LogisticRegression(C=BEST_LOGIT_C, random_state=17, solver='liblinear', max_iter=1000)
    logit_cv_scores = cross_val_score(logit, X_train, y_train, cv=time_split, n_jobs=N_JOBS, scoring='roc_auc', verbose=True)
    print(f"Cross-Validation mean: {logit_cv_scores.mean()}")
    print(f"Cross-Validation std: {logit_cv_scores.std()}")
    logit.fit(X_train, y_train)
    logit_predicted_labels = logit.predict_proba(X_test)[:, 1]

with timer('Performing Catboost Bayesian Optimization'):
    tuned_model = tune_with_bayes(X_train, y_train)
    cboost_predicted_labels = tuned_model.predict_proba(X_test)[:, 1]
    
with timer("Submission"):
    write_to_submission_file(logit_predicted_labels, os.path.join(OUTPUT_DIR, f'logit_{logit_cv_scores.mean()}.csv'))
    write_to_submission_file(cboost_predicted_labels, os.path.join(OUTPUT_DIR, f'cboost_alice_subm_bayes.csv'))
    
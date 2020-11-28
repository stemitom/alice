from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from hyperopt.pyll.stochastic import sample
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import numpy as np
from time import time()

param_gridsearch = {
    'clf__learning_rate' : [0.01, 0.1, 1],
    'clf__max_depth' : [5, 10, 15],
    'clf__n_estimators' : [5, 20, 35], 
    'clf__num_leaves' : [5, 25, 50],
    'clf__boosting_type': ['gbdt', 'dart'],
    'clf__colsample_bytree' : [0.6, 0.75, 1],
    'clf__reg_lambda': [0.01, 0.1, 1],
}

param_randomsearch = {
    'clf__learning_rate': list(np.logspace(np.log(0.01), np.log(1), num = 500, base=3)),
    'clf__max_depth': list(range(5, 15)),
    'clf__n_estimators': list(range(5, 35)),
    'clf__num_leaves': list(range(5, 50)),
    'clf__boosting_type': ['gbdt', 'dart'],
    'clf__colsample_bytree': list(np.linspace(0.6, 1, 500)),
    'clf__reg_lambda': list(np.linspace(0, 1, 500)),
}

lgb_param_space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'max_depth': scope.int(hp.quniform('max_depth', 5, 15, 1)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 5, 51, 1)),
    'num_leaves': scope.int(hp.quniform('num_leaves', 5, 30, 1)),
    'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
}

xgb_param_space = {
    'max_depth' : hp.choice('max_depth', range(5, 30, 1)),
    'learning_rate' : hp.quniform('learning_rate', 0.01, 0.5, 0.01),
    'n_estimators' : hp.choice('n_estimators', range(20, 205, 5)),
    'gamma' : hp.quniform('gamma', 0, 0.50, 0.01),
    'min_child_weight' : hp.quniform('min_child_weight', 1, 10, 1),
    'subsample' : hp.quniform('subsample', 0.1, 1, 0.01),
    'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1.0, 0.01)
}

cb_param_space = {
    'n_estimators': hp.choice('n_estimators', np.arange(50, 250, 25)),
    'max_depth': hp.choice('max_depth', np.arange(5, 11)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
    'reg_lambda': hp.uniform('reg_lambda', 1, 10),
    'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 50)
}


def coarse_search(estimator, parameters, X_train, y_train, optimizer='grid_search', n_iter=None, scoring='accuracy', verbose=True):
    start = time()
    if optimize == 'grid_search':
        grid_obj = GridSearchCV(estimator=estimator, param_grid=parameters, cv=5, 
                                    refit=True, return_train_score=False, scoring=scoring, verbose=True)
    elif optimizer == 'random_search':
        grid_obj = RandomizedSearchCV(estimator=estimator, param_distributions=parameters, cv=5, n_iter=n_iter, 
                                    refit=True, return_train_score=False, scoring='accuracy', random_state=17, verbose=True)
    else: return 'Enter Search Method'
    
    local_best_estimator = grid_obj.best_estimator_
    cvs = cross_val_score(local_best_estimator, X_train, y_train, cv=5, scoring=scoring)
    results = pd.DataFrame(grid_obj.cv_results_)

    print(f"Best Score: {grid_obj.best_score_}")
    print(f"Best parameters: {grid_obj.best_params_}")
    print(f"Cross Validation score mean: {cvs.mean()}")
    print(f"Cross Validation score std: {cvs.std()}")
    print(f"No of parameters combined: {result.shape[0]}")
    print(f"Time Elapsed: {time() - start}")
    return results, local_best_estimator


def hp_param_tuner(param_space, model, X_train, y_train, num_eval, cv=5, scoring='accuracy'):
    start = time()

    def objective_function(params):
        classifier = model(**params)
        cv_score = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=cv, scoring=scoring)
        cv_score_mean = cv_score.mean()    
        print(f"CV mean: {cv_score_mean}")
        return {'loss':1-cv_score_mean, 'status': STATUS_OK }
    
    trials = Trials()
    best_param = fmin(objective_function, param_space, algo=tpe.suggest, 
                            max_evals=num_eval, trials=trials, rstate=np.random.RandomState(17))
    loss = [x['result']['loss'] for x in trials.trials]
    # best_param_values = [x for x in best_param.values()]
    # boosting_type = 'gbdt' if best_param_values[0] == 0 else 'dart'
    if model == LGBMClassifier:
        classifier_best = model(learning_rate=best_param['learning_rate'],
                                num_leaves = int(best_param['num_leaves']),
                                max_depth = int(best_param['max_depth']),
                                n_estimators = int(best_param['n_estimators']),
                                boosting_type = best_param['boosting_type'],
                                colsample_bytree = best_param['colsample_bytree'],
                                reg_lambda = best_param['reg_lambda'])
    elif model == XGBClassifier:
        classifier_best = model(n_estimators = int(best_param['n_estimators']),
                                max_depth = int(best_param['max_depth']),
                                learning_rate = best_param['learning_rate'],
                                gamma = best_param['gamma'],
                                min_child_weight = best_param['min_child_weight'],
                                subsample = best_param['subsample'],
                                colsample_bytree = best_param['colsample_bytree'])
    elif model == CatBoostClassifier:
        classifier_best = model(n_estimators = int(best_param['n_estimators']),
                                max_depth = int(best_param['max_depth']),
                                learning_rate = best_param['learning_rate'],
                                reg_lambda = best_param['reg_lambda'],
                                scale_pos_weight = best_param['scale_pos_weight'])
    classifier_best.fit(X_train, y_train)

    print(f"Best Score: {-1 * min(loss)}")
    print(f"Best parameters: {best_param}")
    print(f"Time Elapsed: {time() - start}")
    print(f"No of parameters combined: {num_eval}")

    return classifier_best, best_param

lgb_clf, lgb_best_param = hp_param_tuner(lgb_param_space, LGBMClassifier, X_train, y_train, 75, 5, 'roc_auc')
xgb_clf, xgb_best_param = hp_param_tuner(xgb_param_space, XGBClassifier, X_train, y_train, 75, 5, 'roc_auc')
cb_clf, xgb_best_param = hp_param_tuner(xgb_param_space, CatBoostClassifier, X_train, y_train, 75, 5, 'roc_auc')

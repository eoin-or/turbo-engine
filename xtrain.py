import pandas as pd
import numpy as np
import category_encoders as ce

from math import sqrt
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from bayes_opt import BayesianOptimization
from catboost import cv, CatBoostRegressor, Pool

train_df = pd.read_csv('training.csv')

X = train_df.drop('Instance', axis=1)
X = X.drop('Income in EUR', axis=1)
y = train_df['Income in EUR']

X_pred = pd.read_csv('test.csv')
X_pred = X_pred.drop('Income', axis=1)
X_pred = X_pred.drop('Instance', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.1)

ct = ColumnTransformer(transformers=[('num_imp', SimpleImputer(strategy='median'), [0, 2, 4, 9]), ('cat_imp', SimpleImputer(strategy='most_frequent'), [1, 3, 5, 6, 7, 8])], remainder='passthrough')

ct.fit(X_train, y_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

jobs = X_train[:,6]
senior_job_terms = ['senior', 'manager', 'doctor', 'lawyer', 'analyst', 'programmer', 'specialist', 'supervisor', 'chief']
senior_job = []
for j in jobs:
    found=False
    for s in senior_job_terms:
        if s in j:
            senior_job.append('yes')
            found = True
            break
    if not found:
        senior_job.append('no')

X_train = np.column_stack((X_train, senior_job))
train_pool = Pool(X_train, label=y_train)

jobs = X_test[:,6]
senior_job = []
for j in jobs:
    found=False
    for s in senior_job_terms:
        if s in j:
            senior_job.append('yes')
            found = True
            break
    if not found:
        senior_job.append('no')

X_test = np.column_stack((X_test, senior_job))
test_pool = Pool(X_test, label=y_test) 

def cat_hyp(depth, l2_leaf_reg, learning_rate, n_estimators):
    params = {'eval_metric': 'RMSE',
            'verbose': False,
            'depth': int(depth),
            'n_estimators': int(n_estimators),
            'l2_leaf_reg': l2_leaf_reg,
            'learning_rate': learning_rate,
            'border_count': 254,
            'use_best_model': True,
            'od_type': 'Iter'
            }

    scores = cv(train_pool, params, fold_count=3)
    return -1 * np.min(scores['test-RMSE-mean'])

bounds = {'depth': (6, 10),
        'l2_leaf_reg': (1, 9),
        'n_estimators': (300, 1000),
        'learning_rate': (0.01, 0.1)}

optimiser = BayesianOptimization(cat_hyp, bounds)
optimiser.maximize(init_points=10, n_iter=50)
params = optimiser.max['params']
params['depth'] = int(params['depth'])
params['n_estimators'] = int(params['n_estimators'])

tuned_model = CatBoostRegressor(optimiser.max['params'])
tuned_model.fit(train_pool)


X_test = enc.transform(X_test)
# X_test = xgb.DMatrix(X_test)
predicted_scores = model.predict(X_pred)

print('\a')
with open('tcd ml 2019-20 income prediction submission file.csv', 'wb') as f:
        index = 111994
        f.write(b'Instance,Income\n')
        for p in predicted_scores:
            f.write(str(index).encode())
            f.write(b',')
            f.write(str(p).encode())
            f.write(b'\n')
            index += 1

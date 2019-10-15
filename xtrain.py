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
train_pool = Pool(X_train, label=y_train, cat_features = [4, 5, 6, 7, 8, 9, 10])

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
test_pool = Pool(X_test, label=y_test, cat_features = [4, 5, 6, 7, 8, 9, 10]) 

def cat_hyp(learning_rate, depth, l2_leaf_reg, random_strength, bagging_temperature):
    params = {'iterations': 700,
            'eval_metric': 'RMSE',
            'verbose': False,
            'depth': int(depth),
            'l2_leaf_reg': l2_leaf_reg,
            'random_strength': random_strength,
            'bagging_temperature': bagging_temperature,
            'use_best_model': True,
            'od_type': 'Iter'
            }

    scores = cv(train_pool, params, fold_count=5)
    return -1 * np.max(scores['test-rmse-mean'])

bounds = {'learning_rate': (0.03, 0.1),
        'depth': (4, 10),
        'l2_leaf_reg': (1, 9),
        'random_strength' : (0, 8),
        'bagging_temperature' : (0, 10)}

optimiser = BayesianOptimization(cat_hyp, bounds)
optimiser.maximize(init_points=10, n_iter=50)
print(optimiser.params)

# enc = ce.TargetEncoder(cols=[4, 5, 6, 7, 8, 9, 10]).fit(X_train, y_train)
# X_train = enc.transform(X_train)

# X_test = enc.transform(X_test)
# X_test = xgb.DMatrix(X_test)
predicted_scores = model.predict(X_test)

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

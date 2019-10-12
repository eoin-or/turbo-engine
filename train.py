import pandas as pd
import numpy as np
import category_encoders as ce
import xgboost as xgb

from math import sqrt
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from bayes_opt import BayesianOptimization
train_df = pd.read_csv('training.csv')

X_train = train_df.drop('Instance', axis=1)
X_train = X_train.drop('Income in EUR', axis=1)
y_train = train_df['Income in EUR']

X_test = pd.read_csv('test.csv')
X_test = X_test.drop('Income', axis=1)
X_test = X_test.drop('Instance', axis=1)

ct = ColumnTransformer(transformers=[('num_imp', SimpleImputer(strategy='median'), [0, 2, 4, 9]), ('cat_imp', SimpleImputer(strategy='most_frequent'), [1, 3, 5, 6, 7, 8])], remainder='passthrough')

ct.fit(X_train, y_train)
X_train = ct.transform(X_train)

jobs = X_train[:,6]
senior_job_terms = ['senior', 'manager', 'doctor', 'lawyer', 'analyst', 'programmer', 'specialist', 'supervisor', 'chief']
senior_job = np.zeros((len(jobs), 1))
temp = []
for j in jobs:
    for s in senior_job_terms:
        if s in j:
            temp.append(True)
            continue
        temp.append(False)
                                                    
X_train = np.column_stack((X_train, senior_job))

enc = ce.TargetEncoder(cols=[4, 5, 6, 7, 8, 9, 10]).fit(X_train, y_train)
X_train = enc.transform(X_train)

dtrain = xgb.DMatrix(X_train, label=y_train)
"""
# clf = GridSearchCV(estimator=RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_split=4, n_jobs=-1), param_grid={'min_samples_split':[3,4]}, cv=5, verbose=2, n_jobs=-1)
grid = {"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
         "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
          "min_child_weight" : [ 1, 3, 5, 7 ],
           "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
            "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }

clf = RandomizedSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state = 4, verbosity=1, gamma=0.0, subsample=0.7, min_child_weight=3, learning_rate=0.1, max_depth=6), param_distributions=grid, verbose=1, n_jobs=-1, cv=10, n_iter=20)

scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
rmse_scores = [sqrt(x * -1) for x in scores]

print("Scores: ", rmse_scores, "\tAverage = ",
              sum(rmse_scores) / len(rmse_scores))
"""
def xgb_evaluate(max_depth, gamma, colsample_bytree, subsample):
    params = {'eval_metric': 'rmse',
            'max_depth': int(max_depth),
            'eta': 0.1,
            'subsample': subsample,
            'gamma': gamma,
            'colsample_bytree': colsample_bytree}

    cv_result = xgb.cv(params, dtrain, num_boost_round=1000, nfold=5)
    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]

xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 7),
    'gamma': (0, 1),
    'colsample_bytree': (0.3, 0.9),
    'subsample': (0.5, 0.9)})

xgb_bo.maximize(init_points=10, n_iter=20, acq='ei')

params = xgb_bo.max['params']
params['max_depth'] = int(params['max_depth'])

model = xgb.train(params, dtrain, num_boost_round=250)

X_test = ct.transform(X_test)

jobs = X_test[:,6]
senior_job = np.zeros((len(jobs), 1))
temp = []
for j in jobs:
    for s in senior_job_terms:
        if s in j:
            temp.append(True)
            continue
        temp.append(False)

X_test = np.column_stack((X_test, senior_job))
X_test = enc.transform(X_test)

X_test = xgb.DMatrix(X_test)
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

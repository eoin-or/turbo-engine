import pandas as pd
import numpy as np
import category_encoders as ce
import xgboost as xgb

from math import sqrt
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

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


# clf = GridSearchCV(estimator=RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_split=4, n_jobs=-1), param_grid={'min_samples_split':[3,4]}, cv=5, verbose=2, n_jobs=-1)
grid={'learning_rate':[0.5, 0.1, 0.15, 0.2, 0.3], 'max_depth':[3, 5, 6]}

clf = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state = 4, verbosity=1, gamma=0.0, subsample=0.5, min_child_weight=7), param_grid=grid, verbose=1, n_jobs=-1, cv=5)

scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
rmse_scores = [sqrt(x * -1) for x in scores]

print("Scores: ", rmse_scores, "\tAverage = ",
              sum(rmse_scores) / len(rmse_scores))

clf.fit(X_train, y_train)
print(clf.best_params_)

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

predicted_scores = clf.predict(X_test)

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


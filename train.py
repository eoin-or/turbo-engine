import pandas as pd
import numpy as np
import category_encoders as ce
import xgboost as xgb

from math import sqrt
from math import isnan

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression, ElasticNetCV, SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_approximation import Nystroem

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

enc = ce.TargetEncoder(cols=[4, 5, 6, 7, 8, 9]).fit(X_train, y_train)
X_train = enc.transform(X_train)


# clf = GridSearchCV(estimator=RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_split=4, n_jobs=-1), param_grid={'min_samples_split':[3,4]}, cv=5, verbose=2, n_jobs=-1)
grid={"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
         "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
          "min_child_weight" : [ 1, 3, 5, 7 ],
           "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
            "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }
clf = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state = 4, verbosity=0, gamma=1, eta=0.01, subsample=0.5), param_grid=grid, verbose=0, n_jobs=-1, cv=5)

scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
rmse_scores = [sqrt(x * -1) for x in scores]

print("Scores: ", rmse_scores, "\tAverage = ",
              sum(rmse_scores) / len(rmse_scores))

clf.fit(X_train, y_train)
print(clf.best_params_)

X_test = ct.transform(X_test)
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


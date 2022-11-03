# %% Imports
from cmath import isnan
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
#import scikit
from skopt import BayesSearchCV
#import scikit-optimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
df = pd.read_csv("df.csv")
test_sample = pd.read_csv("test_set.csv")
# %%
bdd_dummies = pd.get_dummies(data=df, columns = ['group','brand','city'], drop_first=True)
bdd_dummies_X = bdd_dummies.loc[:, bdd_dummies.columns != "price"].copy()
bdd_dummies_X_red = bdd_dummies_X.loc[:, bdd_dummies_X.columns != "language"].copy()
y = bdd_dummies.loc[:,'price'].copy()
X_train = bdd_dummies.loc[:,bdd_dummies.columns != "price"].copy()
X_train_red = X_train.loc[:,X_train.columns != "language"].copy()
Y_train = bdd_dummies.loc[:,"price"].copy()

# %%
def TransformTestSample(test_sample):
    res = test_sample.copy()
    res["request_number"] = np.NaN
    res["request_id"] = res["city"] + res["avatar_id"].astype(str)
    for k in range(1,max(res["order_requests"]) + 1):
        rq_id = res.loc[res["order_requests"]==k,"request_id"].unique()[0]
        max_rq = np.nanmax(res.loc[(res.request_id==rq_id) & (res.order_requests<=k),"request_number"])
        if np.isnan(max_rq):
            max_rq = 0
        res.loc[(res.request_id==rq_id) & (res.order_requests==k),"request_number"] = max_rq + 1
    return(res.drop(['request_id', 'index', 'order_requests','avatar_id'], axis=1))

# %%
df_test = TransformTestSample(test_sample)
features_hotels = pd.read_csv("features_hotels.csv")
merged_df_test = df_test.merge(features_hotels, on = ["hotel_id","city"]).drop('hotel_id', axis=1)
df_test_dummies = pd.get_dummies(data=merged_df_test, columns = ['group','brand','city'], drop_first=True)
df_test_dummies_red = df_test_dummies.loc[:, df_test_dummies.columns != 'language']
# col_names = X_train.columns[np.logical_not(X_train.columns.isin(df_test_dummies.columns))]
# for name in col_names:
#     print(name)
#     df_test_dummies[name] = 0

# %% Random Forest Regressor
rf = RandomForestRegressor().fit(X_train, Y_train)
y_pred_rf = rf.predict(df_test_dummies)
#submission = pd.DataFrame({"index" : df_test_dummies.index, "price" : y_pred_rf})
#submission.to_csv('submission.csv', index=False)

# %% MLP Regressor
mlp = MLPRegressor().fit(X_train, Y_train)
y_pred_mlp = mlp.predict(df_test_dummies)
#submission = pd.DataFrame({"index" : df_test_dummies.index, "price" : y_pred_mlp})
#submission.to_csv('submission.csv', index=False)
# %% Tune RF
# Number of trees in random forest
#n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
n_estimators = [int(x) for x in np.linspace(400, 500, num = 2)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
# %% RF best params
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                               n_iter = 50, cv = 3, verbose=2, random_state=42, 
                               n_jobs = -1)
result = rf_random.fit(X_train, Y_train)
rf_random.best_params_

# %% MLP best params

parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant','adaptive'],
    'max_iter' : [100]
}

mlp = MLPRegressor()
mlp_random = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
mlp_random.fit(X_train, Y_train)

# %% LASSO
clf = linear_model.Lasso()
clf.fit(X_train, Y_train)
print(clf.coef_)

# %% Tune XGBoost
opt = BayesSearchCV(
     xgb.XGBRegressor(objective='reg:squarederror',
                       booster = 'gbtree'),
     {
         'n_estimators': [100, 400, 800],
         'max_depth': [3, 6, 9],
         'learning_rate': [0.05, 0.1, 0.20],
         'min_child_weight': [1, 10, 100]
    },
    n_iter=32,
    random_state=0
 )

opt.fit(X_train, Y_train)
# %% XGBoost
xgboost = xgb.XGBRegressor(objective='reg:squarederror',
                           booster = 'gbtree',
                           n_estimators = 800,
                           max_depth=3, max_leaves=0, min_child_weight=1,
                           learning_rate=0.2).fit(X_train_red, Y_train)
y_pred_xgboost = xgboost.predict(df_test_dummies_red)

submission = pd.DataFrame({"index" : df_test_dummies_red.index, "price" : y_pred_xgboost})
submission.to_csv('submission.csv', index=False)

# %% Cross-Validation XGboost
data_dmatrix = xgb.DMatrix(data=bdd_dummies_X_red,label=y)
params = {"objective":'reg:squarederror','colsample_bytree': 0.3,'learning_rate': 0.2,
                'max_depth': 3, 'alpha': 10,  'min_child_weight':1}

xgb_cv = xgb.cv(dtrain=data_dmatrix, params=params, nfold=5,
                    num_boost_round = 500, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)
# %%

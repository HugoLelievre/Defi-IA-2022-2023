# %%
from cmath import isnan
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
df = pd.read_csv("df.csv")
test_sample = pd.read_csv("test_set.csv")
# %%
bdd_dummies = pd.get_dummies(data=df, columns = ['group','brand','city', 'language'], drop_first=True)
X_train = bdd_dummies.loc[:,bdd_dummies.columns != "price"].copy()
Y_train = bdd_dummies.loc[:,"price"].copy()

# %%
def TransformTestSample(test_sample):
    res = test_sample.copy()
    res["request_number"] = np.NaN
    res["request_id"] = res["city"] + res["language"] + res["avatar_id"].astype(str)
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
df_test_dummies = pd.get_dummies(data=merged_df_test, columns = ['group','brand','city', 'language'], drop_first=True)
# col_names = X_train.columns[np.logical_not(X_train.columns.isin(df_test_dummies.columns))]
# for name in col_names:
#     print(name)
#     df_test_dummies[name] = 0

# %%
rf = RandomForestRegressor().fit(X_train, Y_train)
y_pred_rf = rf.predict(df_test_dummies)
submission = pd.DataFrame({"index" : df_test_dummies.index, "price" : y_pred_rf})
submission.to_csv('submission.csv', index=False)

# %%
mlp = MLPRegressor().fit(X_train, Y_train)
y_pred_mlp = mlp.predict(df_test_dummies)
submission = pd.DataFrame({"index" : df_test_dummies.index, "price" : y_pred_mlp})
submission.to_csv('submission.csv', index=False)
# %%
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
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
# %%
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, Y_train)

# %%

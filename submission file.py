# %%
from cmath import isnan
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV,cross_val_score
from sklearn.neural_network import MLPRegressor
df = pd.read_csv("df.csv")
test_sample = pd.read_csv("final_test_df.csv")
# %%
X_train = df.loc[:,df.columns != "price"].copy()
Y_train = df.loc[:,"price"].copy()

# %%
rf = RandomForestRegressor(n_estimators=400, 
                           min_samples_split=10,
                           min_samples_leaf=2,
                           max_features="sqrt",
                           max_depth=15,
                           bootstrap=False).fit(X_train.drop(['price_group','parking','mobile','request_number'],axis=1), Y_train)
y_pred_rf = rf.predict(test_sample.drop(['price_group','parking','mobile','request_number'],axis=1))
submission = pd.DataFrame({"index" : test_sample.index, "price" : y_pred_rf})
submission.to_csv('submission.csv', index=False)
# %%
pd.DataFrame({"variable":X_train.columns, "importance":rf.feature_importances_}).sort_values("importance", ascending=False)
# %%
clf = RandomForestRegressor(max_depth=8)
CVscore = cross_val_score(clf, X_train, Y_train, cv=5, scoring='neg_mean_squared_error')
print(np.mean(CVscore))
# %%

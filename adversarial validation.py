# %%
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import cv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

pricing_requests = pd.read_csv("new_pricing_requests.csv")
test_sample = pd.read_csv("test_set.csv").drop(["index","order_requests"], axis=1)
train = pricing_requests.loc[(pricing_requests.avatar_nb_requests < 5)].reset_index(drop=True)

# %% AddRequestOrder
def AddRequestOrder(df, request_order = 1):
    data = df.copy()
    data["request_order"] = np.nan
    data["avatar_nb_requests"] = np.nan
    avatar_id = data.loc[0,"avatar_id"]
    date = data.loc[0,"date"]
    city = data.loc[0,"city"]
    language = data.loc[0,"language"]
    mobile = data.loc[0,"mobile"]
    avatar_nb_requests = 1
    for k in range(len(data)):
        if data.loc[k,"avatar_id"] != avatar_id :
            avatar_id = data.loc[k,"avatar_id"]
            date = data.loc[k,"date"]
            city = data.loc[k,"city"]
            language = data.loc[k,"language"]
            mobile = data.loc[k,"mobile"]
            avatar_nb_requests = 1
            request_order += 1
        elif data.loc[k,"date"] != date or data.loc[k,"city"] != city or data.loc[k,"language"] != language or data.loc[k,"mobile"] != mobile:
            date = data.loc[k,"date"]
            city = data.loc[k,"city"]
            language = data.loc[k,"language"]
            mobile = data.loc[k,"mobile"]
            avatar_nb_requests += 1
            request_order += 1
        data.loc[k,"request_order"] = request_order
        data.loc[k,"avatar_nb_requests"] = avatar_nb_requests
    return data
# %%
test = AddRequestOrder(test_sample)

# select only the numerical features
X_test  = test.select_dtypes(include=['number']).copy()
X_train = train.select_dtypes(include=['number']).copy()

# drop the target column from the training data
X_train = X_train.drop(['price'], axis=1)

# add the train/test labels
X_train["AV_label"] = 0
X_test["AV_label"]  = 1

# make one big dataset
all_data = pd.concat([X_train, X_test], axis=0, ignore_index=True)
# %%
# shuffle
all_data_shuffled = all_data.sample(frac=1)

# create our DMatrix (the XGBoost data structure)
X = all_data_shuffled.drop(['AV_label'], axis=1)
X = X.drop(['avatar_id','request_order','hotel_id'], axis=1)
y = all_data_shuffled['AV_label']
XGBdata = xgb.DMatrix(data=X,label=y)

# our XGBoost parameters
params = {"objective":"binary:logistic",
          "eval_metric":"logloss",
          'learning_rate': 0.05,
          'max_depth': 5, }
# %%
# perform cross validation with XGBoost
cross_val_results = cv(dtrain=XGBdata, params=params, 
                       nfold=5, metrics="auc", 
                       num_boost_round=200,early_stopping_rounds=20,
                       as_pandas=True)

# print out the final result
print((cross_val_results["test-auc-mean"]).tail(1))
# %%
classifier = XGBClassifier(eval_metric='logloss',use_label_encoder=False)
classifier.fit(X, y)
fig, ax = plt.subplots(figsize=(12,4))
plot_importance(classifier, ax=ax)
plt.show()
# %%
prediction_test = classifier.predict(test[X.columns])
# %% Stats
from scipy import stats

features_list = X_test.columns.values.tolist()
for feature in features_list:
    statistic, p_value = stats.kstest(X_train[feature], X_test[feature])
    print("KS test value: %.3f" %statistic, "with a p-value %.2f" %p_value, "for the feature",feature)

for feature in features_list:
    statistic, p_value = stats.kstest(X_train[feature], X_test[feature])
    if statistic > 0.1 and p_value < 0.05:
        print("KS test value: %.3f" %statistic, "with a p-value %.2f" %p_value, "for the feature",feature)
# %% MatrixLanguagesCities
def MatrixLanguagesCities(df):
    list_languages = df.language.unique()
    list_cities = df.city.unique()
    mat_cities_languages = pd.DataFrame(np.zeros((len(list_languages),len(list_cities))),
                                                index = list_languages, columns=list_cities)
    for k in range(len(df)):
        individu = df.loc[k]
        mat_cities_languages.loc[individu.language, individu.city] += 1
    return mat_cities_languages

mat_cities_languages_test = MatrixLanguagesCities(test_sample)
mat_cities_languages_train = MatrixLanguagesCities(train.reset_index(drop=True))

# %% kmeans
from sklearn.cluster import KMeans
features_hotels = pd.read_csv("features_hotels.csv")
features_hotels_dummies = pd.get_dummies(features_hotels.drop(['group','hotel_id'],axis=1), columns = ['brand','city'], drop_first=True)
model_kmeans = KMeans(n_clusters = 100)

res = pd.DataFrame()
res['hotel_id'] = features_hotels.hotel_id
res['cluster'] = model_kmeans.fit(features_hotels_dummies).labels_
# %% dummies
bdd_hotel_test = pricing_requests.loc[(pricing_requests.avatar_nb_requests < 5)&(pricing_requests.hotel_id.isin(test_sample.hotel_id.unique()))].reset_index(drop=True)
train_dummies = pd.get_dummies(bdd_hotel_test.drop(['avatar_id','request_order'],axis=1), columns = ['hotel_id','city','language'], drop_first=True)
X_train_dummies = train_dummies.loc[:,train_dummies.columns != "price"]
Y_train_dummies = train_dummies.loc[:,"price"]
# %%
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor().fit(X_train_dummies, Y_train_dummies)
# %%
test_sample = pd.read_csv("test_set.csv")
df_test = AddRequestOrder(test_sample)
test_dummies = pd.get_dummies(data=df_test, columns = ['language','city','hotel_id'], drop_first=True)

test_pred = test_dummies[X_train_dummies.columns]
y_pred_rf = rf.predict(test_pred)
submission = pd.DataFrame({"index" : test_dummies.index, "price" : y_pred_rf})
submission.to_csv('submission.csv', index=False)
# %% ApplyBinClass
def ApplyBinClass(df_train, df_test):
    train = df_train.copy().sample(frac=1).reset_index(drop=True)
    test = df_test.copy().sample(frac=1).reset_index(drop=True)
    price = train.price
    train = train.drop("price", axis=1)
    train['nb_class_test'] = 0
    train["AV_label"] = 0
    test["AV_label"]  = 1
    k = 0
    n = len(test)
    while k + n < len(train):
        X_train = train.loc[k:k+n,train.columns != "nb_class_test"]
        all_data = pd.concat([X_train, test[X_train.columns]], axis=0, ignore_index=True)
        all_data_shuffled = all_data.sample(frac=1)
        X = all_data_shuffled.drop(['AV_label'], axis=1)
        # X = X.drop(['avatar_id','request_order','hotel_id'], axis=1)
        y = all_data_shuffled['AV_label']
        classifier = XGBClassifier(eval_metric='logloss',use_label_encoder=False)
        classifier.fit(X, y)
        fig, ax = plt.subplots(figsize=(12,4))
        plot_importance(classifier, ax=ax)
        plt.show()
        prediction_test = classifier.predict(train[X.columns])
        train['nb_class_test'] += prediction_test
        k += n
    train["price"] = price
    return train.drop("AV_label", axis=1)
        
# %%
hotel_price_mean = train.groupby('hotel_id').price.mean()
hotel_price_std = train.groupby('hotel_id').price.std()
train_temp = train.merge(hotel_price_mean, on='hotel_id', how='left', suffixes=("", "_mean"))
train_temp = train_temp.merge(hotel_price_std, on='hotel_id', how='left', suffixes=("", "_std"))
test_temp = test.copy()
test_temp["price"] = 0
test_temp = test_temp.merge(hotel_price_mean, on='hotel_id', how='left', suffixes=("", "_mean"))
test_temp = test_temp.merge(hotel_price_std, on='hotel_id', how='left', suffixes=("", "_std")).drop("price",axis=1)
# %%
# X_test  = test.select_dtypes(include=['number']).copy().drop(['avatar_id','request_order'],axis=1)
# X_train = train.select_dtypes(include=['number']).copy().drop(['avatar_id','request_order'],axis=1)
# df_bin_class = ApplyBinClass(X_train, X_test)

train_mean_std = train_temp.drop(['avatar_id','request_order','hotel_id'],axis=1)
test_mean_std = test_temp.drop(['avatar_id','request_order','hotel_id'],axis=1)

train_dummies = pd.get_dummies(train_mean_std, columns = ['city','language'], drop_first=True)
test_dummies = pd.get_dummies(test_mean_std, columns = ['city','language'], drop_first=True)
df_bin_class = ApplyBinClass(train_dummies, test_dummies)

# %%
n_class = 1
X_train_rf = df_bin_class.loc[df_bin_class.nb_class_test > n_class, df_bin_class.columns != "price"].drop("nb_class_test", axis=1)
Y_train_rf = df_bin_class.loc[df_bin_class.nb_class_test > n_class, "price"]

X_train_rf = train_dummies.loc[:, train_dummies.columns != "price"]
Y_train_rf = train_dummies.loc[:,"price"]
# %%
old_pricing_requests = pd.read_csv("pricing_requests.csv")
old_train_temp = old_pricing_requests.merge(hotel_price_mean, on='hotel_id', how='left', suffixes=("", "_mean"))
old_train_temp = old_train_temp.merge(hotel_price_std, on='hotel_id', how='left', suffixes=("", "_std"))
old_train_mean_std = old_train_temp.drop(['avatar_id','request_order','hotel_id'],axis=1)
old_train_dummies = pd.get_dummies(old_train_mean_std, columns = ['city','language'], drop_first=True)
# %%
# X_train_rf = X_train_rf.drop("price_std", axis=1)
rf = RandomForestRegressor()
rf.fit(X_train_rf, Y_train_rf)
y_pred_rf = rf.predict(old_train_dummies[X_train_rf.columns])
MSE_rf = np.mean((old_train_dummies.price-y_pred_rf)**2)
print(MSE_rf)
# %%
y_pred_rf = rf.predict(test_dummies[X_train_rf.columns])
submission = pd.DataFrame({"index" : test_dummies.index, "price" : y_pred_rf})
submission.to_csv('submission.csv', index=False)
# %%
plt.figure()
for i in range(844):
    dates = test_sample.loc[test_sample.avatar_id == i, "date"].unique()
    plt.plot(dates)

# %%

# %%
import numpy as np
import pandas as pd
pricing_requests = pd.read_csv("pricing_requests.csv")
features_hotels = pd.read_csv("features_hotels.csv")

# %% Merged_df
merged_df = pricing_requests.merge(features_hotels, on = ["hotel_id","city"])
merged_df.to_csv('merged_df.csv', index=False)
df = merged_df[["price","stock","city","date","language","mobile",
                "request_number", "group","brand","parking",
                "pool","children_policy"]].copy()
# %% ScaleList
def ScaleList(list):
    xmin = min(list) 
    xmax=max(list)
    for i, x in enumerate(list):
        list[i] = (x-xmin) / (xmax-xmin)
# %%
list_hotel = merged_df.hotel_id.unique()
list_hotel_price = []
for hotel_id in list_hotel:
    sub_df = merged_df[(merged_df.request_number == 1) & (merged_df.hotel_id == hotel_id)]
    list_hotel_price.append(np.mean(sub_df.price))

merged_df.hotel_id = pd.Categorical(merged_df.hotel_id)
ScaleList(list_hotel_price)
mean_prices = pd.DataFrame({"price_hotel":list_hotel_price,"hotel_id":list_hotel})
merged_df = merged_df.merge(mean_prices, on = "hotel_id")
# %%
from sklearn.ensemble import RandomForestRegressor
bdd = merged_df[["price","stock","date","language","mobile","request_number","price_hotel"]]
merged_df_dummies = pd.get_dummies(data=bdd, columns = ['language'], drop_first=True)
X_all = merged_df_dummies.loc[:,merged_df_dummies.columns != "price"].copy()
Y_all = merged_df_dummies.loc[:,"price"].copy()
test_sample = pd.read_csv("test_set.csv")
df_test = TransformTestSample(test_sample)
df_test["index_test"] = df_test.index
df_test = df_test.merge(mean_prices, on = "hotel_id")
rf = RandomForestRegressor().fit(X_all, Y_all)
df_test_pred = pd.get_dummies(data=df_test.sort_values("index_test").drop(["hotel_id","index_test","city"],axis=1), columns = ["language"], drop_first=True).reset_index(drop=True)
y_pred_rf = rf.predict(df_test_pred)
# %%
submission = pd.DataFrame({"index" : df_test.index, "price" : y_pred_rf})
submission.to_csv('submission.csv', index=False)
# %%
list_cities = ["amsterdam", "copenhagen", "madrid", "paris", "rome", "sofia", "valletta", "vienna", "vilnius"]
list_cities_price = []
for city in list_cities:
    sub_df = df[(df.request_number == 1) & (df.city == city)]
    list_cities_price.append(np.mean(sub_df.price))

df.city = pd.Categorical(df.city)
ScaleList(list_cities_price)
df["price_city"] = df.city.cat.rename_categories(list_cities_price)

list_brands = df.brand.unique()
list_brands_price = []
for brand in list_brands:
    sub_df = df[(df.request_number == 1) & (df.brand == brand)]
    list_brands_price.append(np.mean(sub_df.price))

df.brand = pd.Categorical(df.brand)
ScaleList(list_brands_price)
df["price_brand"] = df.brand.cat.rename_categories(list_brands_price)

list_groups = df.group.unique()
list_groups_price = []
for group in list_groups:
    sub_df = df[(df.request_number == 1) & (df.group == group)]
    list_groups_price.append(np.mean(sub_df.price))

df.group = pd.Categorical(df.group)
ScaleList(list_groups_price)
df["price_group"] = df.group.cat.rename_categories(list_groups_price)
# %%
df.drop(['brand','group','city','language'],axis=1).to_csv('df.csv', index=False)
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
#%%
test_sample = pd.read_csv("test_set.csv")
df_test = AddRequestOrder(test_sample)
merged_df_test = df_test.merge(hotel_price, on='hotel_id', how='left')
test_dummies = pd.get_dummies(data=merged_df_test.drop(["hotel_id", "mobile", "avatar_id", "stock", "request_order", "order_requests"], axis=1), columns = ['language','city'], drop_first=True).rename(columns={"price" : "price_mean"})

test_pred = test_dummies[X_train.columns]
y_pred_rf = rf.predict(test_pred)
submission = pd.DataFrame({"index" : test_dummies.index, "price" : y_pred_rf})
submission.to_csv('submission.csv', index=False)
# %%
merged_df_test.city = pd.Categorical(merged_df_test.city)
merged_df_test["price_city"] = merged_df_test.city.cat.rename_categories(list_cities_price)
merged_df_test.brand = pd.Categorical(merged_df_test.brand)
merged_df_test["price_brand"] = merged_df_test.brand.cat.rename_categories(list_brands_price)
merged_df_test.group = pd.Categorical(merged_df_test.group)
merged_df_test["price_group"] = merged_df_test.group.cat.rename_categories(list_groups_price)
# %%
merged_df_test.drop(['brand','group','city'],axis=1).to_csv("final_test_df.csv", index=False)


# %%
df_simple = pricing_requests.copy()
hotel_price = df_simple.groupby('hotel_id').price.mean()
df_minus = df_simple.merge(hotel_price, on='hotel_id', how='left', suffixes=("", "_mean")).loc[:,["price","price_mean","date","city","language","avatar_nb_requests","hotel_id"]]
# %%
from sklearn.ensemble import RandomForestRegressor
# trainX, trainY = df_minus.loc[:,df_minus.columns != "price"], df_minus.loc[:,"price"]
# rf = RandomForestRegressor()
# rf.fit(trainX, trainY)
# %%
test_sample = pd.read_csv("test_set.csv")
df_test = TransformTestSample(test_sample).loc[:,["date","hotel_id","request_number"]]
# df_test = pd.read_csv("test_set.csv").loc[:,["date","hotel_id"]]
df_minus_test = df_test.merge(hotel_price, on='hotel_id', how='left').rename(columns={"price": "price_mean"}).drop("hotel_id",axis=1)
# %%
y_pred_rf = rf.predict(df_minus_test.drop("request_number",axis=1))
y_pred_rf = y_pred_rf + df_minus_test.request_number -1
submission = pd.DataFrame({"index" : df_minus_test.index, "price" : y_pred_rf})
submission.to_csv('submission.csv', index=False)
# %%
from sklearn.model_selection import train_test_split 
bdd_dummies = pd.get_dummies(data=df_minus.drop(["hotel_id"], axis=1), columns = ['language','city'], drop_first=True)
# X_train = bdd_dummies.loc[:,bdd_dummies.columns != "price"]
# Y_train = bdd_dummies.loc[:,"price"]
data_train, data_test = train_test_split(bdd_dummies,test_size=0.25,random_state=1)
X_train = data_train.loc[:,data_train.columns != "price"]
Y_train = data_train.loc[:,"price"]
X_test = data_test.loc[:,data_train.columns != "price"]
Y_test = data_test.loc[:,"price"]

rf = RandomForestRegressor().fit(X_train, Y_train)
y_pred_rf = rf.predict(X_test)
MSE_rf = np.mean((Y_test-y_pred_rf)**2)
print(MSE_rf)
# %%
import matplotlib.pyplot as plt
import seaborn as sns
df = pricing_requests[(pricing_requests.hotel_id==995)&(pricing_requests.request_number==1)]

a,b,c = np.polyfit(df.date, df.price, deg = 2)
x = np.sort(df.date.unique().astype(float))
f_x = c + b*x + a*(x**2)

plt.plot(x, f_x)
plt.scatter(df.date,df.price, c='b')
plt.show()
# %%
def SubmissionDegree2(train, test):
    res = []
    for k in test.index:
        df_k = train[(train.hotel_id == test.loc[k,"hotel_id"])&(train.request_number==1)]
        a,b,c = np.polyfit(df_k.date, df_k.price, deg = 2)
        x = test.loc[k,"date"]
        f_x = c + b*x + a*(x**2)
        min_k, max_k = min(df_k.price), max(df_k.price)
        if f_x > max_k:
            f_x = max_k
        if f_x < min_k:
            f_x = min_k
        res.append(f_x)
    return pd.DataFrame({"index":test.index,"price":res})
        
# %%
df_submission = SubmissionDegree2(pricing_requests, test_sample)
df_submission.to_csv('submission.csv', index=False)
# %%
def SubmissionMeanDay(train, test):
    res = []
    for k in test.index:
        df_k = train[(train.hotel_id == test.loc[k,"hotel_id"])&(train.request_number==1)]
        x = test.loc[k,"date"]
        list_jours = [k for k in range(x-5,x+6)]
        f_x = np.mean(df_k.loc[df_k.date.isin(list_jours), "price"])
        res.append(f_x)
    return pd.DataFrame({"index":test.index,"price":res})
# %%
df_submission = SubmissionMeanDay(pricing_requests, test_sample)
df_submission.to_csv('submission.csv', index=False)

# %%
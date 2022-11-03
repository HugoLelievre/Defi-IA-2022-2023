# %%
import numpy as np
import pandas as pd
pricing_requests = pd.read_csv("pricing_requests.csv")
features_hotels = pd.read_csv("features_hotels.csv")

# %%
merged_df = pricing_requests.merge(features_hotels, on = ["hotel_id","city"])
merged_df.to_csv('merged_df.csv', index=False)
df = merged_df[["price","stock","city","date","language","mobile",
                "request_number", "group","brand","parking",
                "pool","children_policy"]].copy()
# %%
list_cities = ["amsterdam", "copenhagen", "madrid", "paris", "rome", "sofia", "valletta", "vienna", "vilnius"]
list_cities_price = []
for city in list_cities:
    sub_df = df[(df.request_number == 1) & (df.city == city)]
    list_cities_price.append(np.mean(sub_df.price))

df.city = pd.Categorical(df.city)
df["price_city"] = df.city.cat.rename_categories(list_cities_price)

list_brands = df.brand.unique()
list_brands_price = []
for brand in list_brands:
    sub_df = df[(df.request_number == 1) & (df.brand == brand)]
    list_brands_price.append(np.mean(sub_df.price))

df.brand = pd.Categorical(df.brand)
df["price_brand"] = df.brand.cat.rename_categories(list_brands_price)

list_groups = df.group.unique()
list_groups_price = []
for group in list_groups:
    sub_df = df[(df.request_number == 1) & (df.group == group)]
    list_groups_price.append(np.mean(sub_df.price))

df.group = pd.Categorical(df.group)
df["price_group"] = df.group.cat.rename_categories(list_groups_price)
# %%
df.drop(['brand','group','city','language'],axis=1).to_csv('df.csv', index=False)
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
test_sample = pd.read_csv("test_set.csv")
df_test = TransformTestSample(test_sample).drop('language',axis=1)
features_hotels = pd.read_csv("features_hotels.csv")
merged_df_test = df_test.merge(features_hotels, on = ["hotel_id","city"]).drop('hotel_id', axis=1)

# %%
merged_df_test.city = pd.Categorical(merged_df_test.city)
merged_df_test["price_city"] = merged_df_test.city.cat.rename_categories(list_cities_price)
merged_df_test.brand = pd.Categorical(merged_df_test.brand)
merged_df_test["price_brand"] = merged_df_test.brand.cat.rename_categories(list_brands_price)
merged_df_test.group = pd.Categorical(merged_df_test.group)
merged_df_test["price_group"] = merged_df_test.group.cat.rename_categories(list_groups_price)
# %%
merged_df_test.drop(['brand','group','city'],axis=1).to_csv("final_test_df.csv", index=False)


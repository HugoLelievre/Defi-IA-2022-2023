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

# %%
def calc_smooth_mean(df, by, on, m):
    # Compute the global mean
    mean = df[on].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + m * mean) / (counts + m)

    # Replace each value by the according smoothed mean
    return df[by].map(smooth)

df['x_0'] = calc_smooth_mean(df, by='x_0', on='y', m=10)
# %%
import numpy as np
import pandas as pd
pricing_requests = pd.read_csv("pricing_requests.csv")
features_hotels = pd.read_csv("features_hotels.csv")

# %%
merged_df = pricing_requests.merge(features_hotels, on = ["hotel_id","city"])
merged_df.to_csv('merged_df.csv', index=False)
df = merged_df[["price","stock","hotel_id","city","date","language","mobile",
                "request_number", "group","brand","parking",
                "pool","children_policy"]].copy()
df.to_csv('df.csv', index=False)
# %%

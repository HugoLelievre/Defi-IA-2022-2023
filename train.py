import pandas as pd
from sklearn.ensemble import RandomForestRegressor

pricing_requests = pd.read_csv("pricing_requests.csv")
df_train = pricing_requests.loc[pricing_requests.avatar_nb_requests < 5]
hotel_price = df_train.groupby('hotel_id').price.mean()
df_train = df_train.merge(hotel_price, on='hotel_id', how='left', suffixes=("", "_mean"))
hotel_date = df_train.groupby('hotel_id').date.mean()
df_train = df_train.merge(hotel_date, on='hotel_id', how='left', suffixes=("", "_mean"))
data_train = pd.get_dummies(data=df_train.loc[:,["price","price_mean","date_mean","language","date","stock","city"]], columns = ['language','city'], drop_first=True)

X_train = data_train.loc[:,data_train.columns != "price"]
Y_train = data_train.loc[:,"price"]

model_rf = RandomForestRegressor().fit(X_train, Y_train)
# y_pred_rf = model_rf.predict(X_test)
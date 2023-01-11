print(" --- Data gathering --- ")

import gradio as gr
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 
import pickle

filename = "rfmodel.pickle"
loaded_model = pickle.load(open(filename, "rb"))
mean_price = pd.read_csv("price_hotel_id.csv", index_col=[0])
mean_date = pd.read_csv("date_hotel_id.csv", index_col=[0])

def price_mean_comput(hotel_id):
    if mean_price.index.isin([hotel_id]).any():
        return mean_price[mean_price.index == hotel_id].price[hotel_id]
    else:
        return mean_price.mean().price

def date_mean_comput(hotel_id):
    if mean_date.index.isin([hotel_id]).any():
        return mean_date[mean_date.index == hotel_id].date[hotel_id]
    else:
        return mean_date.mean().date
        
def list_cities():
    cities = ['city_amsterdam', 'city_copenhagen', 'city_madrid', 'city_paris', 'city_rome', 'city_sofia', 'city_valletta', 'city_vienna','city_vilnius']
    cities_df = pd.DataFrame(data=np.zeros(shape=(1,len(cities)), dtype=int), columns = cities)
    return cities_df

def list_languages():
    languages = ['language_austrian', 'language_belgian', 'language_bulgarian', 'language_croatian', 'language_cypriot', 'language_czech', 'language_danish', 'language_dutch', 'language_estonian', 'language_finnish', 'language_french', 'language_german', 'language_greek', 'language_hungarian', 'language_irish', 'language_italian', 'language_latvian', 'language_lithuanian', 'language_luxembourgish', 'language_maltese', 'language_polish', 'language_portuguese', 'language_romanian', 'language_slovakian', 'language_slovene', 'language_spanish', 'language_swedish']
    languages_df = pd.DataFrame(data=np.zeros(shape=(1,len(languages)), dtype=int), columns = languages)
    return languages_df

def predict(hotel_id, date, stock, city, language):
    cities_df = list_cities()
    languages_df = list_languages()
    city = 'city_' + city
    language = 'language_' + language
    cities_df[city] = 1
    languages_df[language] = 1
    x_pred = pd.DataFrame({'price_mean':[price_mean_comput(hotel_id)],'date_mean':[date_mean_comput(hotel_id)],'date':[date],'stock':[stock]})
    x_pred = pd.concat([x_pred, languages_df, cities_df], axis = 1)
    print(x_pred.columns)
    return loaded_model.predict(x_pred)
choice = 1000
print(" --- Data gathered --- \n")
while choice != 1 or choice != 2:
    print("You have 2 options :")
    print("1_ Train the model (RandomForestRegressor) and evaluate it on the test data.")
    print("2_ Launch the grad.io application.")
    choice = input("Select your option ")
    if choice == "1":
        print(" --- Beginning of the training --- ")
        pricing_requests = pd.read_csv("pricing_requests.csv")
        df_train = pricing_requests.loc[pricing_requests.avatar_nb_requests < 5]
        hotel_price = df_train.groupby('hotel_id').price.mean()
        df_train = df_train.merge(hotel_price, on='hotel_id', how='left', suffixes=("", "_mean"))
        hotel_date = df_train.groupby('hotel_id').date.mean()
        df_train = df_train.merge(hotel_date, on='hotel_id', how='left', suffixes=("", "_mean"))
        data_train = pd.get_dummies(data=df_train.loc[:,["price","price_mean","date_mean","language","date","stock","city"]], columns = ['language','city'], drop_first=False)
        data_train, data_test = train_test_split(data_train,test_size=0.25,random_state=1)
        X_train = data_train.loc[:,data_train.columns != "price"]
        Y_train = data_train.loc[:,"price"]
        X_test = data_test.loc[:,data_train.columns != "price"]
        Y_test = data_test.loc[:,"price"]
        model_rf = RandomForestRegressor().fit(X_train, Y_train)
        print(" --- End of the training --- ")
        y_pred_rf = model_rf.predict(X_test)
        MSE_rf = np.mean((Y_test-y_pred_rf)**2)
        print("Error on training set : " +str(MSE_rf))
    if choice =="2":
        demo = gr.Interface(fn = predict,
                                inputs = [gr.Number(0, label="hotel_id (integer between 0 and 998)"),
                                          gr.Number(0,label ="date (integer between 0 and 44)"),
                                          gr.Number(0,label ="stock (integer)"),
                                          gr.Dropdown(['amsterdam', 'copenhagen', 'madrid', 'paris', 'rome', 'sofia', 'valletta', 'vienna','vilnius'], label = 'city'),
                                          gr.Dropdown(['austrian', 'belgian', 'bulgarian', 'croatian', 'cypriot', 'czech', 'danish', 'dutch', 'estonian', 'finnish', 'french', 'german', 'greek', 'hungarian', 'irish', 'italian', 'latvian', 'lithuanian', 'luxembourgish', 'maltese', 'polish', 'portuguese', 'romanian', 'slovakian', 'slovene', 'spanish', 'swedish'], label = "language")],
                                outputs = "text")
        demo.launch(share=True)


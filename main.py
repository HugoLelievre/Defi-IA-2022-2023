import gradio as gr
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import pickle

filename = "lr.pickle"
loaded_model = pickle.load(open(filename, "rb"))

def list_id_hotel():
    list_id = np.empty(0)
    for i in range(999):
        list_id = np.append(list_id, 'hotel_id_' + str(i))
    hotel_id = pd.DataFrame(data = np.zeros(shape=(1,len(list_id)), dtype=int), columns = list_id)
    return hotel_id

def list_cities():
    cities = ['city_amsterdam', 'city_copenhagen', 'city_madrid', 'city_paris', 'city_rome', 'city_sofia', 'city_valletta', 'city_vienna','city_vilnius']
    cities_df = pd.DataFrame(data=np.zeros(shape=(1,len(cities)), dtype=int), columns = cities)
    return cities_df

def list_languages():
    languages = ['language_austrian', 'language_belgian', 'language_bulgarian', 'language_croatian', 'language_cypriot', 'language_czech', 'language_danish', 'language_dutch', 'language_estonian', 'language_finnish', 'language_french', 'language_german', 'language_greek', 'language_hungarian', 'language_irish', 'language_italian', 'language_latvian', 'language_lithuanian', 'language_luxembourgish', 'language_maltese', 'language_polish', 'language_portuguese', 'language_romanian', 'language_slovakian', 'language_slovene', 'language_spanish', 'language_swedish']
    languages_df = pd.DataFrame(data=np.zeros(shape=(1,len(languages)), dtype=int), columns = languages)
    return languages_df

def predict(date, mobile, hotel, city, language):
    hotel_id = list_id_hotel()
    cities_df = list_cities()
    languages_df = list_languages()
    id_hotel = 'hotel_id_' + str(int(hotel))
    city = 'city_' + city
    language = 'language_' + language
    hotel_id[id_hotel] = 1
    cities_df[city] = 1
    languages_df[language] = 1
    x_pred = pd.DataFrame({'date':[date],'mobile':[int(mobile)]})
    x_pred = pd.concat([x_pred, hotel_id, cities_df, languages_df], axis = 1)
    return loaded_model.predict(x_pred)

demo = gr.Interface(fn = predict,
                   inputs = [gr.Number(0,label ="date (integer between 0 and 44)"),
                             gr.Checkbox(False,label ="Use of a mobile phone to do the request"),
                             gr.Number(0,label ="hotel_id (de 1 à 998 (est-ce qu'il manque pas 999 ?) à voir comment faire)"),
                             gr.Dropdown(['amsterdam', 'copenhagen', 'madrid', 'paris', 'rome', 'sofia', 'valletta', 'vienna','vilnius'], label = 'city'),
                             gr.Dropdown(['austrian', 'belgian', 'bulgarian', 'croatian', 'cypriot', 'czech', 'danish', 'dutch', 'estonian', 'finnish', 'french', 'german', 'greek', 'hungarian', 'irish', 'italian', 'latvian', 'lithuanian', 'luxembourgish', 'maltese', 'polish', 'portuguese', 'romanian', 'slovakian', 'slovene', 'spanish', 'swedish'], label = "language"),
                            ],
                        outputs = "text")
demo.launch(share=True)

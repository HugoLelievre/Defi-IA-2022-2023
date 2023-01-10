# Defi-IA-2022-2023

All our approach to develop the best possible model to predict the price of hotels is in the file analysis.ipynb

The final model we have chosen uses a random forest. We use the quantitative variables date, stock, date_mean and price_mean which respectively correspond to the average date/price of the rows in our database for a given hotel. We also use the city and language variables with one-hot encoding.

# Use of Dockerfile

To launch the gradio.app using a Docker image, apply the following commands. 

1. `git clone https://github.com/HugoLelievre/Defi-IA-2022-2023.git`
This will clone the repository at the desired location on your computer.

2. `docker build -t "name_of_image":latest .`
This will build a Docker image from Dockerfile.

3. `docker run -it "name_of_image":latest`
It will run your image on a Docker container. `-it` will allow you to enter in Interactive mode and to launch the grad.io app.

# About all the files

- Dockerfile contains all the informations to build the Docker image
- Requetes defi IA.ipynb contains the code we used to do the requests on the API
- analysis.ipynb contains our analysis about the models we tried to use and importances of the various features 
- date_hotel_id.csv contains informations about mean_date according to the hotel_id
- feature_hotels.csv contains informations about the hotels
- main.py contains the code used by the Docker image to run the grad.io app
- price_hotel_id.csv contains informations about mean_price according to the hotel_id
- pricing_requests.csv contains all the requests we made from the API
- rf_model.pickle contains our final model
- test_set.csv is a test set
- train.py is the file used to train the Random forest model


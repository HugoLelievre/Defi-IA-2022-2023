# Defi-IA-2022-2023

All our approach to develop the best possible model to predict the price of hotels is in the file analysis.ipynb

The final model we have chosen uses a random forest. We use the quantitative variables date, stock, date_mean and price_mean which respectively correspond to the average date/price of the rows in our database for a given hotel. We also use the city and language variables with one-hot encoding.

# Use of Dockerfile

To launch the gradio.app using a Docker image, apply the following procedure. 

The easiest way to use our Docker image is to use it from a gcloud instance. The 3 following steps will show you how do set up one.

1. Create a VM instance on Google Cloud following [this tutorial](https://wikistat.github.io/AI-Frameworks/gcloud_set_up.html) 

2. Activate the instance.

3. Click on the arrow next to `SSH` on the instances page and display the gcloud command. Copy this command and run it into a local terminal. This will launch a remote gcloud terminal. The command should looke like `gcloud compute ssh --zone "zone_of_your_instance" "name_of_your_instance"  --project "true-server-xxxxxx"`.

Now, you are in the terminal of your gcloud instance. The following command will make the model working.

1. `git clone https://github.com/HugoLelievre/Defi-IA-2022-2023.git`
This will clone the repository in your gcloud instance.

2. `cd Defi-IA-2022-2023`
This will move you to the directory where files are on your gcloud instance.

3. `wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10foNNbiUnDcFQ-h-KAkOqnED2JOZ2PWW' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10foNNbiUnDcFQ-h-KAkOqnED2JOZ2PWW" -O rfmodel.pickle && rm -rf /tmp/cookies.txt` This will download the file containing the model we trained to directly use the gradio application and put in in a file called `rfmodel.pickle`.

4. `sudo docker build -t image_name:latest .`
This will build a Docker image from Dockerfile. 

5. `sudo docker run -it --name container_name image_name:latest`
It will run your image on a Docker container. `-it` will allow you to enter in Interactive mode and to launch the grad.io app.

6. The, you will have a choice. Typing `1` will train a RandomForestRegressor on the data and return the MSE between the predictions of the model on the test set and the real prices. Typing `2` will launch the grad.io application. The easiest way to use it is to wait for the public URL to appear, and then copy-paste it into your browser to make predictions about hotel prices according to various parameters.

/!\ To stop the connexion to the gcloud instance, do `ctrl + d` on the terminal of your gcloud instance. /!\

/!\ Do not forget to stop your gcloud instance from the instances page after use. /!\

# About all the files

- Dockerfile contains all the informations to build the Docker image
- Requetes defi IA.ipynb contains the code we used to do the requests on the API
- analysis.ipynb contains our analysis about the models we tried to use and importances of the various features 
- date_hotel_id.csv contains informations about mean_date according to the hotel_id
- feature_hotels.csv contains informations about the hotels
- main.py contains the code used by the Docker image to run the grad.io app
- price_hotel_id.csv contains informations about mean_price according to the hotel_id
- pricing_requests.csv contains all the requests we made from the API
- test_set.csv is a test set
- train.py is the file used to train the Random forest model
- As explained previously, you need to download a trained model from [this Google Drive](https://drive.google.com/file/d/10foNNbiUnDcFQ-h-KAkOqnED2JOZ2PWW/view?usp=share_link) using the command 3. This file needs to be called `rfmodel.pickle` and to be in the directory containing all the previous files.


# Defi-IA-2022-2023

Le notebook Requêtes défi IA permet d'effectuer les requêtes qu'on veut et de les ajouter dans le fichier pricing_requests.csv

Le fichier python include meta data permet de fusionner le fichier pricing_requests.csv avec les méta données qu'on possède sur les hôtels, grâce à la colonne "hotel_id" présente dans les 2 fichiers.

Ce fichier python enregistre 2 BDD, celle fusionée (merged_df.csv), et la même mais avec uniquement les colonnes qu'on utilise pour l'entraînement (df.csv).
Le Notebook défi IA ML est celui qu'on utilise pour entraîner et faire des tests sur les modèles.

# Use of Dockerfile

To launch the gradio.app using a Docker image, apply the following commands. 

1. `git clone https://github.com/HugoLelievre/Defi-IA-2022-2023.git`
This will clone the repository at the desired location on your computer.

2. `docker build -t "name_of_image":latest .`
This will build a Docker image from Dockerfile.

3. `docker run -it "name_of_image":latest`
It will run your image on a Docker container. `-it` will allow you to enter in Interactive mode and to launch the gradio.app.

4. `python main.py`
It will run the Python code and give you the link to the gradio.app.

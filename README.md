# Defi-IA-2022-2023

Le notebook Requêtes défi IA permet d'effectuer les requêtes qu'on veut et de les ajouter dans le fichier pricing_requests.csv

Le fichier python include meta data permet de fusionner le fichier pricing_requests.csv avec les méta données qu'on possède sur les hôtels, grâce à la colonne "hotel_id" présente dans les 2 fichiers.

Ce fichier python enregistre 2 BDD, celle fusionée (merged_df.csv), et la même mais avec uniquement les colonnes qu'on utilise pour l'entraînement (df.csv).
Le Notebook défi IA ML est celui qu'on utilise pour entraîner et faire des tests sur les modèles.

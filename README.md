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
It will run your image on a Docker container. `-it` will allow you to enter in Interactive mode and to launch the gradio.app.

4. `python main.py`
It will run the Python code and give you the link to the gradio.app.

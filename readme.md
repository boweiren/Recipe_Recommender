# Recipe Recommender

> Created by Harrison Mitgang, Alina Shabanov, Bowei Ren, Jane Wu.

The recipe recommender allows a user to upload images of ingredients they have to a website, rate a few recipes, and be provided with a list of recipes they would like and have the ingredients to make.

## Deployment Instructions

#### Install Required Packages

The recipe recommender is run using `Python 3.8.5` or a similar version. Ensure you have this runtime.

Next, install python dependencies using pip:

> `$ pip install -r website/requirements.txt`

> `$ pip install notebook`

#### Create Pickled Dataset

Download the Food.com Recipes and Interactions dataset from [Kaggle](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions) and save the zip file as `raw_interactions.zip` in the project's root directory (this directory).

Start a Jupyter server:

> `$ jupyter notebook`

Open `recipe_pickling.ipynb` and run all the cells. This should save 3 pickled datasets to `website/cached_models`. These datasets help with retraining the recommender system. Once this is complete, you can kill the Jupyter Server.

#### Run Server

To start the Flask server driving the web app:

> `$ cd website`

> `$ python main.py`

_Note: This will likely start the server on port `5000`, so you can open the site at [`localhost:5000`](http://localhost:5000)._

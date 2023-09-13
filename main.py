#import librairies
import sklearn
import joblib
import pandas as pd
import numpy as np
from fonctions_maisons import extraire_la_première_lettre
from flask import Flask, request


#from flask_cors import CORS

#Load model
#pipeline = joblib.load("model_final_rf")
#https://towardsdatascience.com/3-ways-to-deploy-machine-learning-models-in-production-cdba15b00e
#Demarrer l'appli Flask
# https://github.com/noirbizarre/flask-restplus/issues/440
#https://stackoverflow.com/questions/19962699/flask-restful-cross-domain-issue-with-angular-put-options-methods
#https://flask-cors.readthedocs.io/en/latest/
app = Flask('__name__')
#cors = CORS(app)
#Page d'acceuil

#Faire de prediction
@app.route("/predict", methods = ["POST"])
def predict():
  df =pd.DataFrame(request.json)
  pipeline = joblib.load("model_final_rf")

  results = pipeline.predict(df)[0]

  return (str(results), 201)

#Test
@app.route("/Ping", methods = ["GET"])
def pong():
  return ("Pong", 200)

@app.route('/')
def index():
  return "<h1>Bienvenue, Utiliser /predict en POST pour faire des predictions</h1>"


# Si on est dans le main, on lance l'api
if __name__ == "__main__":
  app.run(host='0.0.0.0')

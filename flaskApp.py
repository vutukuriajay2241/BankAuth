# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:45:19 2020

@author: Sai Ajay Vutukuri
"""


from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_in_decision_tree = open('decisionTreeClassifier.pkl','rb')
pickle_in_randon_forest= open('randomForestClassifier.pkl','rb')

randomForestClassifier= pickle.load(pickle_in_randon_forest)
decisionTreeClassifier= pickle.load(pickle_in_decision_tree)

@app.route('/')
def welcome():
    return "Welcome"

@app.route('/predict')
def predict_note_authentication():
    
    """Lets Authenticate the Bank Note
    ---
    parameters:
        - name: variance
          in: query
          type: number
          required: true
        - name: skewness
          in: query
          type: number
          required: true
        - name: curtosis
          in: query
          type: number
          required: true
        - name: entropy
          in: query
          type: number
          required: true
    responses:
        200:
            description: The output values
    """
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    pred_randon_forest = randomForestClassifier.predict([[variance,skewness,curtosis,entropy]])
    pred_decision_tree = decisionTreeClassifier.predict([[variance,skewness,curtosis,entropy]])
    return "The predicted values for RandomForrest Model and Decision Tree model are " + str(pred_randon_forest) + ' and ' + str(pred_decision_tree) + " respecteively"

@app.route('/predict_file', methods=['POST'])
def predict_note_authentication_file():
    
    """Lets Authenticate the Bank Note
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true
        
    responses:
        200:
            description: The output values
    """
    df_test = pd.read_csv(request.files.get("file"))
    
    pred_randon_forest = randomForestClassifier.predict(df_test)
    pred_decision_tree = decisionTreeClassifier.predict(df_test)
    return "The predicted values for RandomForrest Model and Decision Tree model are " + str(list(pred_randon_forest)) + ' and ' + str(list(pred_decision_tree)) + " respecteively"


if __name__ == '__main__':
    app.run()

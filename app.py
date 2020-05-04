# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 21:11:33 2020

@author: pavitra
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/predictt',methods=['POST'])
def predictt():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = np.array(int_features)
    prediction = model.predict([final_features])

    output = round(prediction[0], 2)
    
    if output==0:
        output="normal"

    elif output==1:
        output="hypothyroid"
    else:
        output="hyperthyroid"
        
    return render_template('predict.html', prediction_text='You have {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application

## Import ridge and standard scalar pickle

ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
Standard_Scaler = pickle.load(open('models/scalar.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict_data",methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_scaled_data = Standard_Scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_scaled_data)

        return render_template('home.html',result = result[0])
        
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0", port = 8080)


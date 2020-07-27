import numpy as np
import pandas as pd
from flask_cors import  cross_origin
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

@cross_origin()
@app.route('/')
def home():
    return render_template('index.html')
@cross_origin()
@app.route('/predict',methods=['POST'])
def predict():
    Pregnancies = float(request.form['Pregnancies'])
    Glucose = float(request.form['Glucose'])
    BloodPressure = float(request.form['BloodPressure'])
    SkinThickness = float(request.form['SkinThickness'])
    Insulin = float(request.form['Insulin'])
    BMI = float(request.form['BMI'])
    DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
    Age = float(request.form['Age'])

    filename = 'modelForPrediction.sav'
    loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage
    scalar = pickle.load(open("sandardScalar.sav", 'rb'))
    # predictions using the loaded model file
    prediction = loaded_model.predict(scalar.transform(
        [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]))
    print('prediction is', prediction)

    if prediction == [1]:
        res_val = "  Dibeties ! plese Consult a Doctor & take  precautions . "
    else:
        res_val = "No dibetis ! hurry  your healthy . "

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000, debug=True)
	#app.run(debug=True)
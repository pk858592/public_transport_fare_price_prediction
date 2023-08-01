import pickle
from flask import Flask,request,render_template
import numpy as np
import pandas as pd


app = Flask(__name__)

# load the pickle file

model = pickle.load(open("model.pkl","rb"))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input values from form and store in list in correct order
    inputs = [
        request.form['pickup_longitude'],
        request.form['pickup_latitude'],
        request.form['dropoff_longitude'],
        request.form['dropoff_latitude'],
        request.form['passenger_count'],
        request.form['pickup_hour'],
        request.form['pickup_date'],
        request.form['pickup_month'],
        request.form['pickup_day'],
        request.form['distance']
    ]

    # Convert inputs to floats
    inputs = [float(i) for i in inputs]

    # Make prediction and return result
    prediction = model.predict([inputs])
    output = round(prediction[0], 2)
    return render_template("index.html", prediction_text="the fare amount is {}".format(output))
if __name__ == '__main__':
    app.run(debug=True)
#importing all modules
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load("Student_marks_vs_hours_model.pkl")
df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    global df
    input_features = [int(x) for x in request.form.values()]
    features_value = np.array(input_features)
    
    #validating input hours
    if input_features[0] <0 or input_features[0] >24:
        return render_template('index.html', prediction_text='Please Enter valid number of hours(1-24)!!')
        

    output = model.predict([features_value])[0][0].round(4)

    return render_template('index.html', prediction_text='If you study for {} hours per day, You will score approximately {}% marks! '.format( int(features_value[0]),output))


if __name__ == "__main__":
    app.run(debug=True)
    

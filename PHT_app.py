import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
import pickle
import pandas as pd
from urllib.parse import urlencode
#import webbrowser
app = Flask(__name__)

# loading models
# normalized = pickle.load(open('Normalizing.pkl', 'rb'))
lr_model = pickle.load(open('final_model.pkl', 'rb'))
X = pickle.load(open('train_data.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
url = 'http://personalproductivityapp.depaulmysore.in/Modules/Prediction/predict.php'
@app.route('/')
def home():
    return redirect(url)    

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # FOR loop to retrieve values from the form
    new_observation = []
    for value in request.form.values():
        # new_observation = new_observation.append(value)
          # create a new nested list with the value
        new_observation.append(value) 
    new_observation=[new_observation]
    print(new_observation)
    new_observation_df = pd.DataFrame(new_observation, columns=X.columns)
    

# Preprocess the input features using the same preprocessor object used in training
    new_observation_processed = preprocessor.transform(new_observation_df)

# Get predictions from the Linear Regression model
    lr_prediction = lr_model.predict(new_observation_processed)
 
    # input_data = np.empty([0,])
    # for value in request.form.values():
    #     input_data = np.append(input_data, float(value))    
    # mean = normalized['mean']
    # std = normalized['std']
    # normalized_input_data = (input_data - mean) / std
    # predictions = nn_model.predict(normalized_input_data)
    # # Set threshold for binary classification
    # threshold = 0.5
    # binary_predictions = (predictions > threshold).astype(int)
    # Print the binary prediction   
    output = int(lr_prediction)
    print(output)
    #return output   
    # redirect_url = 'http://localhost:8080/Ratan/prediction/predict.php?output=' + str(output)    
    redirect_url = url + "?output=" + str(output)    

    return redirect(redirect_url) 

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')



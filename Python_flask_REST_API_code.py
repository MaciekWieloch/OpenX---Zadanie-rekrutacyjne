from flask import Flask, request
from flask_restful import Api
import joblib
import numpy as np
import json
import os

app = Flask(__name__)
api = Api(app)


# Load the trained models
project_path = os.getcwd()
model_path = os.path.join(project_path, 'models')

heuristic_model = joblib.load(os.path.join(model_path,'Summary_for_heuristics.pkl'))
Decision_tree_model = joblib.load(os.path.join(model_path,'Decision_tree_model.pkl'))
K_NN_model = joblib.load(os.path.join(model_path,'K_NN_model.pkl'))
ANN = joblib.load(os.path.join(model_path,'ANN_model.pkl'))


#Load the scaler
sc = joblib.load(os.path.join(model_path,'scaler_object.pkl'))

# Define the API endpoints
@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the request body
    input_data = request.get_json()

    # Extract the input features and convert them to a numpy array
    features = np.array(input_data['features'])
    features = features.reshape(1,54)


    # Get the model name from the request
    model_name = input_data['model']

    # Use the selected model to make predictions
    if model_name == 'heuristic':

        prediction = np.abs(features[0,0]-heuristic_model.Elevation).argmin()+1

    elif model_name == 'Decision_tree_model':

        features = np.concatenate([sc.transform(features[:, 0:10]), features[:, 10:55]], axis=1)
        prediction = Decision_tree_model.predict(features)

    elif model_name == 'K_NN_model':

        features = np.concatenate([sc.transform(features[:, 0:10]), features[:, 10:55]], axis=1)
        prediction = K_NN_model.predict(features)

    elif model_name == 'ANN_model':

        features = np.concatenate([sc.transform(features[:, 0:10]), features[:, 10:55]], axis=1)
        prediction = ANN.predict(features)
        prediction = prediction.argmax()+1

    # Convert the prediction to a list and return it as a JSON response
    response = {'prediction': prediction.tolist()}
    return json.dumps(response)

if __name__ == '__main__':
    app.run(debug=True)

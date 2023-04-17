import requests
import json

#Example input features (please remember to cut the last value from below examples when coping to features =... (last value is the covertype, that I left in the commented examples to see if the result is correct))
#2880,209,17,216,30,4986,206,253,179,4323,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,2
#2739,117,24,127,53,3281,253,210,71,6033,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,5
#3201,53,7,488,26,3667,224,225,135,4255,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1
#3352,325,13,124,8,5070,188,226,177,3481,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,7
#2390,63,23,124,50,192,233,186,72,932,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3
#2335,33,29,30,4,474,201,165,83,395,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6
#2193,48,7,0,0,1442,223,225,138,553,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4


# Define the input features and the selected model
features = [2335,33,29,30,4,474,201,165,83,395,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

#'heuristic'
#'Decision_tree_model'
#'K_NN_model'
#'ANN_model'
model_name = 'ANN_model'

# Define the URL of the prediction endpoint
url = 'http://localhost:5000/predict'

# Define the request data as a dictionary
data = {
    'features': features,
    'model': model_name
}

# Send the prediction request to the locally running REST API
response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'})

# Parse the JSON response and print the prediction
prediction = json.loads(response.text)['prediction']
print('Prediction:', prediction)

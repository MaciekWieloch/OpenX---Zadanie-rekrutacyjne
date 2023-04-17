import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle


#Get the project directory
project_path = os.getcwd()


#Prepare the DataFrame for modelling (format it to dataframe and give it column names)
data_path = os.path.join(project_path, 'Data')
df = np.loadtxt(os.path.join(data_path,'covtype.data'), delimiter = ',')
colnames = np.loadtxt(os.path.join(data_path,'covtype_colnames.data.txt'), delimiter=",", dtype= str)
DF = pd.DataFrame(data=df,columns=colnames)

#Save the df for further analysis
with open(os.path.join(data_path,'Prepared data','DF.pkl'), 'wb') as f:
    pickle.dump(DF, f)


#Set X(independent) and Y(dependent) variables
X = DF.iloc[:,:-1].values
Y = DF.iloc[:,-1].values


#Split into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=1)


#Feature Scaling (only for quantitative variables)
sc = StandardScaler()
X_train[:,0:10] = sc.fit_transform(X_train[:,0:10])
X_test[:,0:10] = sc.transform(X_test[:,0:10])


#Save the data for ML training
with open(os.path.join(data_path,'Prepared data','X_train.pkl'), 'wb') as f:
    pickle.dump(X_train, f)
with open(os.path.join(data_path,'Prepared data','X_test.pkl'), 'wb') as f:
    pickle.dump(X_test, f)
with open(os.path.join(data_path,'Prepared data','Y_train.pkl'), 'wb') as f:
    pickle.dump(Y_train, f)
with open(os.path.join(data_path,'Prepared data','Y_test.pkl'), 'wb') as f:
    pickle.dump(Y_test, f)


#Save the scaler for future input transformation
model_path = os.path.join(project_path, 'models', 'scaler_object.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(sc, f)

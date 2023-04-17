import numpy as np
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import os
import pickle
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score


#Get the project directory
project_path = os.getcwd()

#Load the preprocessed data
X_train = joblib.load(os.path.join(project_path,'Data','Prepared data','X_train.pkl'))
X_test = joblib.load(os.path.join(project_path,'Data','Prepared data','X_test.pkl'))
Y_train = joblib.load(os.path.join(project_path,'Data','Prepared data','Y_train.pkl'))
Y_test = joblib.load(os.path.join(project_path,'Data','Prepared data','Y_test.pkl'))

#Heuristic model
DF = joblib.load(os.path.join(project_path,'Data','Prepared data','DF.pkl'))

#Create some summary DF to look for a rule of thumb
DF_summary = pd.DataFrame(data=np.zeros((7,56)),columns=np.append(DF.columns,"Cover_Type_weight"))

DF_summary.Cover_Type = np.unique(DF.Cover_Type)
for col in DF.columns[0:10]:
    for tree_cov in np.unique(DF.Cover_Type):
        DF_summary[col][tree_cov-1] = DF[col][DF["Cover_Type"]==tree_cov].mean()

for col in DF.columns[10:54]:
    for tree_cov in np.unique(DF.Cover_Type):
        DF_summary[col][tree_cov-1] = DF[col][DF["Cover_Type"]==tree_cov].sum()

for tree_cov in np.unique(DF.Cover_Type):
    DF_summary["Cover_Type_weight"][tree_cov-1] = DF["Cover_Type"][DF["Cover_Type"]==tree_cov].count()/DF.Cover_Type.count()

#The rule of thumb for heuristic will be: 1.Classify the sample elevation by assigning its elevation to the nearest mean elevation
#Therefore save the summary DF to be loaded in REST API
model_path = os.path.join(project_path, 'models', 'Summary_for_heuristics.pkl')
DF_summary.to_pickle(model_path)



#Train Simple ML Model1 = Decision Tree
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)

#Pred with Simple ML Model 1 = Decision Tree
Y_pred = classifier.predict(X_test)

#Make Confiusion Matrix for Simple ML Model 1 = Decision Tree
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
ac = accuracy_score(Y_test, Y_pred)
print(ac)

#Save the model
model_path = os.path.join(project_path, 'models', 'Decision_tree_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(classifier, f)



#Train Simple ML Model2 = K-NN
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)
classifier.fit(X_train, Y_train)

#Pred with Simple ML Model 2 = K-NN
Y_pred = classifier.predict(X_test)

#Make Confiusion Matrix for Simple ML Model 2 = K-NN
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
ac = accuracy_score(Y_test, Y_pred)
print(ac)

#Save the model
model_path = os.path.join(project_path, 'models', 'K_NN_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(classifier, f)



#Prepare ANN
#Initialize the ANN
ann = tf.keras.models.Sequential()

#Add 1st layer
ann.add(tf.keras.layers.Dense(units=54, activation='relu'))

#Add 2nd layer
ann.add(tf.keras.layers.Dense(units=54, activation='relu'))

#Add output layer
ann.add(tf.keras.layers.Dense(units=7, activation='softmax'))

#Compiling ANN
ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Training ANN
ann.fit(X_train, Y_train-1, batch_size=32, epochs=100)

#Pred on test data
Y_pred = ann.predict(X_test)
accuracy = ann.evaluate(X_test, Y_test-1)

Y_pred_temp = np.zeros(len(Y_pred))
for i in range(len(Y_pred)):
    Y_pred_temp[i] = Y_pred[i].argmax()

Y_pred = Y_pred_temp

#Make Confiusion Matrix for ANN model
cm = confusion_matrix(Y_test, Y_pred+1)
print(cm)
ac = accuracy_score(Y_test, Y_pred+1)
print(ac)

#Save the model
model_path = os.path.join(project_path, 'models', 'ANN_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(ann, f)


"""
Step 1: importing basic libraries required
tensorflow framework version 2.6.0 is needed for running framework and for getting the models and layers in it.
matplotlib used to plot the visualizing diagrams
pandas used for access/alter dataset
sklearn to divide the dataset, scaling the features of dataset
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ReLU
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.metrics import Accuracy
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import BinaryCrossentropy
"""
Step 2: reading the csv dataset
"""

dataset = pd.read_csv("Churn_Modelling_dataset/Churn_Modelling.csv")

""" Step 3: 
Feature engineering:
- dividing the dataset into dependant and independant features 
- one hot encoding of the categorical values is needed in dataset if present
- dropping the categorical columns from the independant dataset
- concatenating the converted one hot encoded columns to the independant dataset
"""

X = dataset.iloc[:,3:13] ### dropping the row no, id and name column as it is not required.
Y = dataset.iloc[:,13]

geography_one_hot_encoded = pd.get_dummies(X['Geography'], drop_first=True)
gender_one_hot_encoded = pd.get_dummies(X['Gender'], drop_first=True)

X = X.drop(['Geography','Gender'], axis=1)

X = pd.concat([X, geography_one_hot_encoded, gender_one_hot_encoded], axis=1)
# print("\n ******** After concatenation the number of input nodes : {}\n".format(X.shape)) ### 11 input nodes

"""
Step 4: splitting converted dataset into train and test data
"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=0)

"""
Step 5: scaling the dataset
"""

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


"""
Step 6: creating the ANN model
- create dense layer of 11 neurons with relu activation(why 11? ---> independant variable columns after feature engineering)
- create hidden dense layers again with relu activation
- creating output dense layer with node 1 with sigmoid activation as it is a binary classification problem
"""

classifier = Sequential()

classifier.add(Dense(units=11, activation=ReLU))
classifier.add(Dense(units=7, activation=ReLU))
classifier.add(Dense(units=6, activation=ReLU))

classifier.add(Dense(units=1, activation='sigmoid'))

"""
Step 7: compiling and training the ANN model with EARLY STOPPING

"""
# classifier.compile(optimizer=Adam, loss=BinaryCrossentropy, metrics=Accuracy)
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

model_history = classifier.fit(X_train, Y_train , validation_split=0.33, batch_size=10, epochs=1000)
"""
Step 1:
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
Step 2:
"""
data = pd.read_csv('./heart.csv')
# print(data.head())

"""
Step 3:
"""
X = data.iloc[:,:13]
y = data.iloc[:,13]

"""
Step 4:
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

"""
Step 5:
"""
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""
Step 6:
"""
classifier = Sequential()
classifier.add(Dense(activation = "relu", input_dim = 13,
					units = 8, kernel_initializer = "uniform"))
classifier.add(Dense(activation = "relu", units = 14,
					kernel_initializer = "uniform"))
classifier.add(Dense(activation = "sigmoid", units = 1,
					kernel_initializer = "uniform"))

"""
Step 7:
"""
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy',
				metrics = ['accuracy'] )

classifier.fit(X_train , y_train , batch_size = 8 ,epochs = 100)

"""
Step 8:
"""
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix: \n{}\n".format(cm)) 

accuracy = (cm[0][0]+cm[1][1])/(cm[0][1] + cm[1][0] +cm[0][0] +cm[1][1]) # accuracy = (TP + TN) / (TP + FN + FP + TN)
print("\nAccuracy of model: ", accuracy*100)


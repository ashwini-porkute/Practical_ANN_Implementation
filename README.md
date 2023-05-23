# Practical_ANN_Implementation
ANN implementation check ANN1.py and ANN2.py scripts with different datasets.  

ANN1.py Script reference taken from "https://www.youtube.com/watch?v=ydzFSLDmHmE&list=PLZoTAELRMXVPiyueAqA_eQnsycC_DSBns&index=6&ab_channel=KrishNaik"
ANN2.py Script reference taken from "https://www.geeksforgeeks.org/heart-disease-prediction-using-ann/"

Steps to be followed while designing ANN:
--------------------------------------------

Step 1: 
Importing basic libraries required
- tensorflow for model api and keras layers requirement.
- matplotlib used to plot the visualizing diagrams
- pandas used for access/alter dataset
- sklearn to divide the dataset, scaling the features of dataset.

Step 2:
Reading the csv dataset (dataset loading)

Step 3: 
Feature engineering which may include the following steps as per requirement:
- dividing the dataset into dependant and independant features 
- one hot encoding of the categorical values is needed if required
- dropping the categorical columns from the independant dataset
- concatenating the converted one hot encoded columns to the independant dataset

Step 4: 
Splitting converted dataset into train and test data

Step 5: 
Scaling the dataset

Step 6: 
Creating the ANN model
- create dense layer of total number of input neurons with relu activation
- create hidden dense layers again with relu activation
- creating output dense layer with node 1 with sigmoid activation as it is a binary classification problem

Step 7: 
Compiling and training the ANN model without EARLY STOPPING.

Step 8:
Performing prediction on trained model and rescaling the results.

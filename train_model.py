# import pandas as pd
# from sklearn.model_selection import train_test_split  #split data into training and testing sets
# from sklearn.neural_network import MLPRegressor  #creating the nn regression model
# from sklearn.preprocessing import StandardScaler  #scaling target values
# from sklearn.metrics import mean_squared_error  #calculating the RMSE of predictions
# #!pip install rdkit
# #!pip install rdkit-pypi
# from rdkit import Chem  #handling chemical data in SMILES format
# from rdkit.Chem import AllChem, MACCSkeys  #generating Morgan fingerprints and MACCS keys
# import numpy as np
# import matplotlib.pyplot as plt

# #Set random seed for reproducibility of results
# np.random.seed(17)

# #Function to generate Morgan fingerprints for a given SMILES string
# def generate_morgan_fingerprints(smiles, radius=2, n_bits=2048):

#     #Convert SMILES to a molecule object
#     molecule = Chem.MolFromSmiles(smiles)

#     #Generate a Morgan fingerprint with the specified radius and number of bits
#     fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=n_bits)

#     #Convert the fingerprint to a NumPy array and return it
#     return np.array(fingerprint)

# #Function to generate MACCS keys for a given SMILES string
# def generate_maccs_keys(smiles):

#     #Convert SMILES to a molecule object
#     molecule = Chem.MolFromSmiles(smiles)

#     #Generate MACCS keys for the molecule
#     fingerprint = MACCSkeys.GenMACCSKeys(molecule)

#     #Convert the fingerprint to a NumPy array and return it
#     return np.array(fingerprint)

# #Apply the functions to create Morgan fingerprints and MACCS keys for each molecule in the dataset
# data['Morgan'] = data['smiles'].apply(generate_morgan_fingerprints)
# data['MACCS'] = data['smiles'].apply(generate_maccs_keys)

# #Extract features (Morgan and MACCS) and target (Exp) values

# #Morgan fingerprints as feature matrix
# X_morgan = np.array(list(data['Morgan']))

# #MACCS keys as feature matrix
# X_maccs = np.array(list(data['MACCS']))

# #Target values, reshaped to be a 2D array
# y = data['exp'].values.reshape(-1, 1)

# #Split the dataset into training and testing sets (80% train, 20% test)
# X_morgan_train, X_morgan_test, y_train, y_test = train_test_split(X_morgan, y, test_size=0.2, random_state=17)
# X_maccs_train, X_maccs_test, _, _ = train_test_split(X_maccs, y, test_size=0.2, random_state=17)

# #Initialize a StandardScaler for standardizing target values
# scaler = StandardScaler()

# #Fit the scaler to the training targets and transform them
# y_train_scaled = scaler.fit_transform(y_train)

# #Transform the test targets using the same scaler (important to use the same scaling)
# y_test_scaled = scaler.transform(y_test)

# #Define and train an MLPRegressor model using Morgan fingerprints
# mlp_morgan = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=17)

# #Fit the model on the scaled training data
# mlp_morgan.fit(X_morgan_train, y_train_scaled.ravel())

# #Define and train an MLPRegressor model using MACCS keys
# mlp_maccs = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=17)

# #Fit the model on the scaled training data
# mlp_maccs.fit(X_maccs_train, y_train_scaled.ravel())

# #Predict target values for the test set using the Morgan model
# y_pred_morgan_scaled = mlp_morgan.predict(X_morgan_test)

# #Predict target values for the test set using the MACCS model
# y_pred_maccs_scaled = mlp_maccs.predict(X_maccs_test)

# #Inverse transform the scaled predictions to the original scale for Morgan fingerprints
# y_pred_morgan = scaler.inverse_transform(y_pred_morgan_scaled.reshape(-1, 1))

# #Inverse transform the scaled predictions to the original scale for MACCS keys
# y_pred_maccs = scaler.inverse_transform(y_pred_maccs_scaled.reshape(-1, 1))

# #Calculate the RMSE for the Morgan fingerprints model
# rmse_morgan = np.sqrt(mean_squared_error(y_test, y_pred_morgan))

# #Calculate the RMSE for the MACCS keys model
# rmse_maccs = np.sqrt(mean_squared_error(y_test, y_pred_maccs))

# #Print the RMSE values for both models
# print(f"RMSE using Morgan fingerprints: {rmse_morgan}")
# print(f"RMSE using MACCS keys: {rmse_maccs}")

# #Create a bar chart to compare RMSE values of the two models because i am a visual learner

# #Store RMSE values in a list
# rmse_values = [rmse_morgan, rmse_maccs]

# #Corresponding labels for the models
# labels = ['Morgan Fingerprints', 'MACCS Keys']

# #Set the figure size for the bar chart
# plt.figure(figsize=(8, 5))

# #Create the bar chart using RMSE values and labels
# plt.bar(labels, rmse_values)

# #Set the label for the y-axis
# plt.ylabel('RMSE')

# #Set the title for the chart
# plt.title('RMSE Comparison of MLPRegressor Models')

# #Display the bar chart
# plt.show()

# #Create scatter plots to visualize the relationship between actual and predicted values for both models
# plt.figure(figsize=(12, 6))

# #Scatter plot for Morgan fingerprints model predictions

# #Create a subplot (1 row, 2 columns, 1st plot)
# plt.subplot(1, 2, 1)

# #Scatter plot with some transparency
# plt.scatter(y_test, y_pred_morgan, alpha=0.5, color='magenta', edgecolor='black')

# #green line for perfect prediction
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'g--', lw=2)

# #Label for x-axis
# plt.xlabel('Actual Values')

# #Label for y-axis
# plt.ylabel('Predicted Values')

# #Title of the plot
# plt.title('Morgan Fingerprints')

# #Scatter plot for MACCS keys model predictions

# #Create a subplot (1 row, 2 columns, 2nd plot)
# plt.subplot(1, 2, 2)

# #Scatter plot with some transparency
# plt.scatter(y_test, y_pred_maccs, alpha=0.5, color='cyan', edgecolor='black')

# #black line for perfect prediction
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'black', lw=2)

# #Label for x-axis
# plt.xlabel('Actual Values')

# #Label for y-axis
# plt.ylabel('Predicted Values')

# #Title of the plot
# plt.title('MACCS Keys')

# #Adjust layout to prevent overlap and display the scatter plots
# plt.tight_layout()
# plt.show()



#### `LICENSE`

# MIT License

# Copyright (c) 2024 Belita :)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#### `src/train_model.py`

#This is the main Python script for training the fingerprint-based model. It should accept two outside hyperparameters (`hidden_layer_sizes` and `max_iter`) and calculate the RMSE, saving the results in `results.txt`.

#```python
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a model using Morgan Fingerprints or MACCS Keys")
parser.add_argument("--hidden_layer_sizes", type=int, nargs='+', default=[100], help="Sizes of hidden layers for MLPRegressor")
parser.add_argument("--max_iter", type=int, default=500, help="Maximum number of iterations for MLPRegressor")
args = parser.parse_args()

# Load dataset
data = pd.read_csv('Lipophilicity.csv')  # Adjust path if necessary
smiles = data['smiles']  # Replace with actual column name for SMILES
y = data['exp']  # Replace with actual target column name

# Function to generate Morgan fingerprints
def generate_morgan_fingerprints(smiles, radius=2, n_bits=2048):
    molecule = Chem.MolFromSmiles(smiles)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=n_bits))

# Function to generate MACCS keys
def generate_maccs_keys(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    return np.array(MACCSkeys.GenMACCSKeys(molecule))

# Apply functions to create fingerprint features
data['Morgan'] = data['smiles'].apply(generate_morgan_fingerprints)
data['MACCS'] = data['smiles'].apply(generate_maccs_keys)

# Extract features and target
X_morgan = np.array(list(data['Morgan']))
X_maccs = np.array(list(data['MACCS']))
y = data['exp'].values.reshape(-1, 1)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_morgan, y, test_size=0.2, random_state=17)

# Initialize scaler and scale target values
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.transform(y_test)

# Define and train MLPRegressor model
mlp_model = MLPRegressor(hidden_layer_sizes=tuple(args.hidden_layer_sizes), max_iter=args.max_iter, random_state=17)
mlp_model.fit(X_train, y_train_scaled.ravel())

# Predict and calculate RMSE
y_pred_scaled = mlp_model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Retrieve conda environment name
conda_env = os.getenv("CONDA_DEFAULT_ENV")

# Save results
with open("results.txt", "w") as f:
    f.write(f"Test RMSE: {rmse:.4f}\n")
    f.write(f"Conda Environment: {conda_env}\n")
    f.write(f"Hyperparameters: hidden_layer_sizes={args.hidden_layer_sizes}, max_iter={args.max_iter}\n")

print(f"Results saved to results.txt with RMSE: {rmse:.4f}")
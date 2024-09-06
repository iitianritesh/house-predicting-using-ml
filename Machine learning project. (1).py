#!/usr/bin/env python
# coding: utf-8

# # Project: Predicting Housing Prices Using Machine Learning
# 
# 

# In[ ]:


#Problem Statement:
#Create a machine learning model to predict housing prices based on various features such as square footage, number of bedrooms, location, etc.

#Step-by-Step Instructions:

#Define the Objective:

#Goal: Predict the price of a house.
#Type of Problem: Regression, since the output is a continuous value.

#Data Collection:

#Collect a dataset with features including house prices, areas in square feet, number of bedrooms, number of bathrooms, location, etc.
#A popular dataset for this project is the Boston Housing Dataset, which can be sourced from libraries like Scikit-learn or UCI Machine Learning Repository.

#Data Preprocessing:

#Data Cleaning: Handle missing values by either removing the rows/columns or imputing them using techniques like median, mean, or mode.
#Feature Selection: Select relevant features that contribute significantly to the target variable (price). Use correlation matrices to examine relationships.
#Feature Engineering: Create new features if necessary (e.g., price per square foot).
#Data Transformation: Normalize or standardize the features to ensure they have a similar scale, especially if you're using models sensitive to feature scales.


#Exploratory Data Analysis (EDA):

#Visualize the relationships between features and the target variable using scatter plots, histograms, and pair plots.
#Understand the distribution and outliers in the data which may affect the model performance.


#Model Selection:

#Choose Algorithms: Consider multiple regression algorithms like Linear Regression, Decision Tree Regression, Random Forest Regression, and Gradient Boosting Machines.
#Train-Test Split: Split the data into training and testing sets, commonly in a 80-20 or 70-30 ratio.



#Model Training:

#Train different models on the training set.
#Use cross-validation techniques like K-Fold to get a better estimate of model performance.


#Model Evaluation:

#Use metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) to evaluate the performance of each model.
#Assess the models on the test set and compare their performance.


#Hyperparameter Tuning:

#Identify the best parameters for each model using techniques like Grid Search or Randomized Search.
#Re-train the models using the best-found parameters and evaluate again.


#Model Deployment:

#Choose the best-performing model.
#Save the model using serialization techniques (e.g., Pickle in Python).
#Create a simple interface (such as a web app using Flask or Django) where users can input features to get predictions.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv("C:/Users/rites/OneDrive/Desktop/data.csv")

# Display the first few rows to understand the data
print(data.head())

# Handle missing values if any
data = data.dropna()

# Define features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=[np.number]).columns

# Preprocessing pipelines for both numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the distribution of the target variable
sns.histplot(y, kde=True)
plt.title('Distribution of Housing Prices')
plt.show()

# Correlation matrix
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
# Example: Create new features if applicable
# Skipping if 'year_built' is not available

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

# Create a pipeline that combines preprocessing and model training
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor())])

# Train the model
model_pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = model_pipeline.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score

# Calculate Mean Squared Error and R2 Score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__learning_rate': [0.01, 0.1, 0.2],
    'regressor__max_depth': [3, 5, 7]
}

# Perform grid search
grid_search = GridSearchCV(estimator=model_pipeline, param_grid=param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f'Best Parameters: {best_params}')
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Save the trained model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Load the trained model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:





# In[ ]:





# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Step 1: Load the cancer dataset
# Breast cancer dataset from sklearn.datasets
cancer_data = load_breast_cancer()

# Step 2: Separate features and target variable
X = cancer_data.data  # Features (independent variables)
y = cancer_data.target  # Target (0 = Malignant, 1 = Benign)

# Step 3: Standardize the feature data
# PCA is affected by the scale of the data, so we scale it to mean=0 and variance=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scaling the data

# Step 4: Perform PCA to reduce the dataset to 2 components
# Principal Component Analysis (PCA) to reduce dimensions to 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 5: Output the explained variance ratio
# Show the amount of variance explained by the 2 PCA components
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by the 2 components: {explained_variance}")

# Step 6: Plot the data points using the first 2 principal components
# Visualization of the two components and their separation of data points
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=80)
plt.title("PCA - 2 Components of Cancer Dataset")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.colorbar(label='Cancer Diagnosis (0: Malignant, 1: Benign)')
plt.show()

# Bonus (Optional): Logistic Regression on PCA-reduced features

# Step 7: Split the PCA-reduced data into training and testing sets
# Train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Step 8: Initialize and train the Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Step 9: Make predictions on the test data
y_pred = log_reg.predict(X_test)

# Step 10: Evaluate the Logistic Regression model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression accuracy with PCA-reduced features: {accuracy}")

# Note: This accuracy reflects how well the logistic regression model performs after PCA.


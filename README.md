#Overview
To underscore the important variables, this project applies Principal Component Analysis (PCA) on the breast cancer dataset from the sklearn.datasets module. It also uses logistic regression for a predictive mode of analysis. Another way is to retrieve features from the dataset and classify the tumors either as malignant or benign.

#Contents
Cancer_Analysis (Main program)
README.md (This README file)
Image of the output

#Requirements
Make sure you have the following libraries installed in your Python environment:

numpy
pandas
matplotlib
seaborn
scikit-learn

#You can install these libraries using pip:

to install type the following command: pip install numpy pandas matplotlib seaborn scikit-learn
Dataset
This project bases its analysis on the breast cancer dataset which is available in sklearn.datasets. It tracked characteristics of tumors and the same tumors as the labels, which categorize them as malignant or benign.

#Running the code
Open the directory in which the code is kept
Run the Python script by typing the following command in the terminal:python cancer_analysis.py

#Expected Output
The script will then perform PCA on the Breast Cancer dataset thus leaving the dataset with only two principal components.
Two PCA components will be taken to capture the variances in the data and the amount of the variances will be echoed out in the terminal.

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# Load the Iris dataset using Pandas
iris_df = pd.read_csv("C:/Users/jnikh/Downloads/Machine Learning Assignment 5/datasets/datasets/Iris.csv")

# Preprocess the dataset using StandardScaler and LabelEncoder from Scikit-learn
from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris_df.iloc[:,range(0,4)].values)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(iris_df['Species'].values)

# Apply Linear Discriminant Analysis to reduce the dimensionality of the dataset to two dimensions
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# Create a DataFrame with the reduced dataset and target variable
data = pd.DataFrame(X_lda)
data['target'] = y
data.columns = ["LD1", "LD2", "target"]

# Plot the two-dimensional dataset using Seaborn and Matplotlib libraries
markers = ['s', 'x', 'o']
colors = ['r', 'b', 'g']
sns.lmplot(x="LD1", y="LD2", data=data, hue='target', markers=markers, fit_reg=False, legend=False)
plt.legend(loc='upper center')
plt.show()


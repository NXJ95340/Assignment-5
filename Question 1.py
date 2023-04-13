#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVC, LinearSVC
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

# Load dataset
credit_card_df = pd.read_csv("C:/Users/jnikh/Downloads/Machine Learning Assignment 5/datasets/datasets/cc.csv")
credit_card_df.head()
credit_card_df['TENURE'].value_counts()

# Select features and target variable
features = credit_card_df.iloc[:,[1,2,3,4]]
target = credit_card_df.iloc[:,-1]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
credit_card_df['CUST_ID'] = le.fit_transform(credit_card_df.CUST_ID.values)

# Perform PCA on unscaled data
pca_unscaled = PCA(n_components=2)
principal_components_unscaled = pca_unscaled.fit_transform(features)

principal_df_unscaled = pd.DataFrame(data = principal_components_unscaled, columns = ['principal component 1', 'principal component 2'])

final_df_unscaled = pd.concat([principal_df_unscaled, credit_card_df[['TENURE']]], axis = 1)
print("PCA on unscaled data:\n", final_df_unscaled.head())

# Perform K-means clustering on unscaled data
n_clusters = 2 # this is the k in kmeans
km_unscaled = KMeans(n_clusters=n_clusters)
km_unscaled.fit(features)

# predict the cluster for each data point
y_cluster_kmeans_unscaled = km_unscaled.predict(features)

# Calculate Silhouette Score on unscaled data
from sklearn import metrics
silhouette_score_unscaled = metrics.silhouette_score(features, y_cluster_kmeans_unscaled)
print("Silhouette Score on unscaled data:", silhouette_score_unscaled)

# Perform PCA on scaled data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

pca_scaled = PCA(n_components=2)
principal_components_scaled = pca_scaled.fit_transform(features_scaled)

principal_df_scaled = pd.DataFrame(data = principal_components_scaled, columns = ['principal component 1', 'principal component 2'])

final_df_scaled = pd.concat([principal_df_scaled, credit_card_df[['TENURE']]], axis = 1)
print("\nPCA on scaled data:\n", final_df_scaled.head())

# Perform K-means clustering on scaled data
n_clusters = 2 # this is the k in kmeans
km_scaled = KMeans(n_clusters=n_clusters)
km_scaled.fit(features_scaled)

# predict the cluster for each data point
y_cluster_kmeans_scaled = km_scaled.predict(features_scaled)

# Calculate Silhouette Score on scaled data
silhouette_score_scaled = metrics.silhouette_score(features_scaled, y_cluster_kmeans_scaled)
print("Silhouette Score on scaled data:", silhouette_score_scaled)


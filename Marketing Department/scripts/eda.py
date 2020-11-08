# -*- coding: utf-8 -*-
"""

@author: bevan
@project: Marketing Department Business Case
    
"""

# --------------------------------------------------------------- #
#### Importing Libraries ####
# --------------------------------------------------------------- #

from keras.optimizers import SGD
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

dataset = pd.read_csv('../data/Marketing_data.csv')

# --------------------------------------------------------------- #
#### EDA ####
# --------------------------------------------------------------- #

dataset.head()
dataset.columns
dataset.describe()
dataset.info()

# --------------------------------------------------------------- #
#### Data Visualization ####
# --------------------------------------------------------------- #

# Check where the null values are and replace them with the mean
sns.heatmap(dataset.isnull(), yticklabels=False, cbar=False, cmap='Blues')

dataset.loc[(dataset['MINIMUM_PAYMENTS'].isnull() == True),
            'MINIMUM_PAYMENTS'] = dataset['MINIMUM_PAYMENTS'].mean()
dataset.loc[(dataset['CREDIT_LIMIT'].isnull() == True),
            'CREDIT_LIMIT'] = dataset['CREDIT_LIMIT'].mean()

dataset.isnull().sum()
dataset.duplicated().sum()

# Drop the columns
dataset.drop('CUST_ID', axis=1, inplace=True)

# KDE Plot
plt.figure(figsize=(10, 50))

for i in range(len(dataset.columns)):
    plt.subplot(17, 1, i + 1)
    sns.distplot(dataset[dataset.columns[i]],
                 kde_kws={'color': 'b', 'lw': 3, 'label': 'KDE'},
                 hist_kws={'color': 'g'})

    plt.title(dataset.columns[i])

plt.tight_layout()

# Correlations Plot
correlations = dataset.corr()

f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(correlations, annot=True)

# --------------------------------------------------------------- #
#### K-Means Alogrithm Method ####
# --------------------------------------------------------------- #

# Scale the data
scaler = StandardScaler()

dataset_scaled = scaler.fit_transform(dataset)

scores_1 = []

range_values = range(1, 20)

for i in range_values:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(dataset_scaled)
    scores_1.append(kmeans.inertia_)

plt.plot(scores_1, 'bx-')
plt.title('Finding the right number of clusters')
plt.xlabel('Clusters')
plt.ylabel('Scores WCSS')
plt.show()

# --------------------------------------------------------------- #
#### K-Means With Market Segamentation ####
# --------------------------------------------------------------- #

kmeans = KMeans(8)
kmeans.fit(dataset_scaled)
labels = kmeans.labels_

cluster_centers = pd.DataFrame(
    data=kmeans.cluster_centers_, columns=[dataset.columns])

cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data=cluster_centers, columns=[dataset.columns])

# Concatenate the cluster labels
dataset_cluster = pd.concat(
    [dataset, pd.DataFrame({'cluster': labels})], axis=1)

# Plot the Histogram of the clusters
for i in dataset.columns:
    plt.figure(figsize=(35, 5))

    for j in range(8):
        plt.subplot(1, 8, j + 1)
        cluster = dataset_cluster[dataset_cluster['cluster'] == j]
        cluster[i].hist(bins=20)
        plt.title('{} \nCluster {}'.format(i, j))

plt.show()

# --------------------------------------------------------------- #
#### Principle Components ####
# --------------------------------------------------------------- #

pca = PCA(n_components=2)
principle_comp = pca.fit_transform(dataset_scaled)

pca_df = pd.DataFrame(data=principle_comp, columns=['pca1', 'pca2'])
pca_df.head()

pca_df = pd.concat([pca_df, pd.DataFrame({'cluster': labels})], axis=1)

plt.figure(figsize=(10, 10))
ax = sns.scatterplot(x='pca1', y='pca2',
                     hue='cluster',
                     data=pca_df, palette=['red', 'green', 'blue', 'pink', 'yellow', 'gray', 'purple', 'black'])

# --------------------------------------------------------------- #
#### Build Model and Train Model (Autoencoder) ####
# --------------------------------------------------------------- #


input_df = Input(shape=(17))

x = Dense(7, activation='relu')(input_df)
x = Dense(500, activation='relu', kernel_initializer='glorot_uniform')(x)
x = Dense(500, activation='relu', kernel_initializer='glorot_uniform')(x)
x = Dense(2000, activation='relu', kernel_initializer='glorot_uniform')(x)

encoded = Dense(10, activation='relu', kernel_initializer='glorot_uniform')(x)

x = Dense(2000, activation='relu',
          kernel_initializer='glorot_uniform')(encoded)
x = Dense(500, activation='relu', kernel_initializer='glorot_uniform')(x)

decoded = Dense(17, kernel_initializer='glorot_uniform')(x)

# Autoencoder
autoencoder = Model(input_df, decoded)

# Encoder Network
encoder = Model(input_df, encoded)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
autoencoder.fit(dataset_scaled, dataset_scaled,
                batch_size=128, epochs=25, verbose=1)

autoencoder.summary()

# Apply the K-Means
pred = encoder.predict(dataset_scaled)

# Optimal the number of Clusters
scores_2 = []

range_values = range(1, 20)

for i in range_values:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(pred)
    scores_2.append(kmeans.inertia_)


plt.plot(scores_2, 'bx-')
plt.title('Finding number of clusters')
plt.xlabel('Clusters')
plt.ylabel('Scores')
plt.show()

# Checking the scores 1 vs scores 2
plt.plot(scores_1, 'bx-', color='r')
plt.plot(scores_2, 'bx-', color='g')

kmeans = KMeans(4)
kmeans.fit(pred)
labels = kmeans.labels_

# Concatenate the cluster labels
dataset_cluster = pd.concat(
    [dataset, pd.DataFrame({'cluster': labels})], axis=1)
dataset_cluster.head()

pca = PCA(n_components=2)
prin_comp = pca.fit_transform(pred)
pca_df = pd.DataFrame(data=prin_comp, columns=['pca1', 'pca2'])

pca_df = pd.concat([pca_df, pd.DataFrame({'cluster': labels})], axis=1)

plt.figure(figsize=(10, 10))

ax = sns.scatterplot(x='pca1', y='pca2',
                     hue='cluster',
                     data=pca_df, palette=['red', 'green', 'blue', 'yellow'])

"""
Assignment 3: Clustering and Classification
Author: Ankit Mehra
Date: 10/14/2023
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import StratifiedShuffleSplit ,train_test_split,cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, silhouette_score, confusion_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import 

# Retrieve and load the Olivetti faces dataset
faces = datasets.fetch_olivetti_faces(shuffle=True, random_state=45)

# decscription of the dataset
print(faces.DESCR)

#standardize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(faces.data)

# Split the training set, a validation set, and a test set using stratified sampling to ensure that there are the same 
# number of images per person in each set.
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=45)
for train_index, test_index in split.split(data_normalized, faces.target):
    X_train, X_test = data_normalized[train_index], data_normalized[test_index]
    y_train, y_test = faces.target[train_index], faces.target[test_index]
    
# Using k-fold cross validation, train a classifier to predict which person is represented in each picture, 
# and evaluate it on the validation set.
model_svc = SVC(kernel='linear', C=1)
scores = cross_val_score(model_svc, X_train, y_train, cv=5)
print("Cross validation mean score: ", scores.mean())

# plot confusion matrix
model_svc.fit(X_train, y_train)
y_pred = model_svc.predict(X_test)
plt.figure(figsize=(10,10))
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#plot image and its predicted label
fig = plt.figure(figsize=(15,10))
for i in range(0, 10):
    ax = fig.add_subplot(1, 10, i+1, xticks=[], yticks=[])
    ax.imshow(X_test[i].reshape(64,64), cmap= 'gray')
    ax.set_title(f"Predicted: {y_pred[i]}\nOrignal: {y_test[i]}")
plt.show()

# Using either Agglomerative Hierarchical Clustering (AHC) or Divisive Hierarchical Clustering
# (DHC) and using the centroid-based clustering rule, reduce the dimensionality of the set by
# using the following similarity measures:
# a) Euclidean Distance [20 points]
# b) Minkowski Distance [20 points]
# c) Cosine Similarity [20 points]

def find_best_n_clusters(metric,data):
    """
    find the best number of clusters and its corresponding silhouette score
    """
    # Define a range of cluster numbers to try
    cluster_range = range(2, 41)

    # Initialize variables to keep track of best silhouette score and corresponding cluster number
    best_score = -1
    best_n_clusters = None
    best_linkage = None

    # Iterate over the range of cluster numbers
    for n_clusters in cluster_range:
        # if metric is euclidean, use ward linkage
        if metric == 'euclidean':
            clustering = AgglomerativeClustering(n_clusters=n_clusters, 
                                                 metric= metric, 
                                                 linkage='ward')
            cluster_labels = clustering.fit_predict(data)
            best_linkage = 'ward'
        else:
            for linkage in ('average', 'complete', 'single'):
                clustering = AgglomerativeClustering(n_clusters=n_clusters,
                                                     metric= metric,
                                                     linkage=linkage)
                cluster_labels = clustering.fit_predict(data)
                
                # Calculate silhouette score
                score = silhouette_score(X=data,labels=cluster_labels, metric=metric)
                
                # Update best score and corresponding cluster number if needed
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
                    best_linkage = linkage

        # Calculate silhouette score
        score = silhouette_score(X=data,labels=cluster_labels, metric=metric)

        # Update best score and corresponding cluster number if needed
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters

    # Print the best number of clusters and its corresponding silhouette score
    print(f"Best number of clusters: {best_n_clusters}")
    print(f"Corresponding silhouette score: {best_score}")
    print(f"Corresponding linkage: {best_linkage}")

find_best_n_clusters('euclidean', data_normalized)
find_best_n_clusters('minkowski', data_normalized)
find_best_n_clusters('cosine', data_normalized)
cosine_model = AgglomerativeClustering(n_clusters=40,
                                       metric='cosine',
                                       linkage='average')
cosine_model.fit(data_normalized)

# Split the training set, and a test set 

X_train_clustered, y_train_clustered, X_test_clustered, y_test_clustered = train_test_split(data_normalized,
                                                                            cosine_model.labels_, 
                                                                            test_size=0.2,
                                                                            random_state=45)
    
# train a classifier to predict which person is represented in each picture, using the reduced dimensionality set
# and evaluate it on the validation set.
model = SVC(kernel='linear', C=1)
scores = cross_val_score(model, X_train_clustered, y_train_clustered, cv=5)
print(scores, scores.mean())
model.fit(X_train_clustered, y_train_clustered)
predicted_after_clustering = model.predict(X_test_clustered)

# accuracy on the orginal dataset
accuracy_score(y_test_clustered, predicted_after_clustering)

# confusion matrix
plt.figure(figsize=(13,10))
sns.heatmap(confusion_matrix(y_test_clustered, predicted_after_clustering), annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

predicted_on_orginal = model.predict(X_test)

# plot image and its predicted label
fig = plt.figure(figsize=(20,20))
for i in range(0, 10):
    ax = fig.add_subplot(1, 10, i+1, xticks=[], yticks=[])
    ax.imshow(X_test[i].reshape(64,64), cmap= 'gray')
    ax.set_title(f"Predicted: {predicted_on_orginal[i]}\nOrignal: {y_test[i]}")
plt.show()

# accuracy on the orginal dataset
print(accuracy_score(y_test, predicted_on_orginal))

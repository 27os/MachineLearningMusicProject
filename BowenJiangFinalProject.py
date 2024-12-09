# -*- coding: utf-8 -*-
"""
Created on Mon May 13 21:35:29 2024

@author: Bowen Jiang
"""

import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer,auc
from sklearn.metrics import roc_curve
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import numpy as np
random.seed(14073479)
np.random.seed(14073479)

import pandas as pd
data = pd.read_csv('musicData.csv')

data = data.dropna()


replacement = {'Major': 1.0, 'Minor': 0.0}
data['mode'] = data['mode'].replace(replacement)

one_hot_encoded = pd.get_dummies(data['key'])
one_hot_encoded = one_hot_encoded.astype(float)
one_hot_encoded.columns = [f'{col}' for col in one_hot_encoded.columns]
data = pd.concat([data, one_hot_encoded], axis=1)



genre_to_float = {
    'Electronic':1.0, 'Anime': 2.0, 'Jazz':3.0, 'Alternative':4.0, 'Country':5.0, 'Rap':6.0, 'Blues':7.0, 'Rock':8.0,
     'Classical':9.0, 'Hip-Hop':10.0
}
data['music_genre'] = data['music_genre'].replace(genre_to_float)





data['tempo'] = data['tempo'].astype(str).str.strip()
data = data[~data['tempo'].str.contains('\?')]
data['tempo'] = pd.to_numeric(data['tempo'], errors='coerce')

unique_artists = data['artist_name'].unique()
unique_artist_count = len(unique_artists)
print(unique_artist_count)

unique_songs = data['track_name'].unique()
unique_song_count = len(unique_songs)
print(unique_song_count)





features = ['popularity', 'acousticness','danceability', 'duration_ms','energy','instrumentalness','liveness','loudness',
            'speechiness','tempo','valence']

for x in features:
    data[x] = (data[x] - data[x].mean()) / data[x].std()

test = []

train_set = pd.DataFrame()
test_set = pd.DataFrame()

genres = data['music_genre'].unique()

for genre in genres:
    
    genre_data = data[data['music_genre'] == genre]
    
    
    genre_test = genre_data.sample(n=500, random_state=42)
    
    
    test_set = pd.concat([test_set, genre_test])
    
    
    genre_train = genre_data.drop(genre_test.index)
    train_set = pd.concat([train_set, genre_train])

print(f"Training Set Size: {train_set.shape[0]}")
print(f"Testing Set Size: {test_set.shape[0]}")

   
X = data.drop(['music_genre','instance_id','artist_name','track_name','obtained_date','key'], axis=1)

common_ids = pd.merge(train_set[['instance_id']], test_set[['instance_id']], on='instance_id', how='inner')
print(f"Number of common IDs: {common_ids.shape[0]}")


X_train = train_set.drop(['music_genre','instance_id','artist_name','track_name','obtained_date','key'], axis=1)

y_train = train_set['music_genre']


X_test = test_set.drop(['music_genre','instance_id','artist_name','track_name','obtained_date','key'], axis=1)

y_test = test_set['music_genre']




tsne = TSNE(n_components=2, perplexity=20, random_state=42, n_iter=300)
tsne_results = tsne.fit_transform(X)


range_n_clusters = list(range(2, 25))  
silhouette_scores = []
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=42,n_init='auto')
    cluster_labels = clusterer.fit_predict(tsne_results)
    silhouette_avg = silhouette_score(tsne_results, cluster_labels)
    silhouette_scores.append(silhouette_avg)


plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()


optimal_clusters = range_n_clusters[silhouette_scores.index(max(silhouette_scores))]
print("Optimal number of clusters:", optimal_clusters)


kmeans = KMeans(n_clusters=optimal_clusters, random_state=42,n_init='auto')
clusters = kmeans.fit_predict(tsne_results)
centers = kmeans.cluster_centers_


total_distance = kmeans.inertia_
print("Total sum of distances of points to their cluster centers:", total_distance)


plt.figure(figsize=(8, 6))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)  
plt.colorbar(scatter)
plt.title('t-SNE and k-Means Clustering')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.show()
 






classes = np.unique(y_test)
y_test_binarized = label_binarize(y_test, classes=classes)
print(y_test_binarized)

model = RandomForestClassifier(random_state=42)


param_grid = {
    'n_estimators': [300],
    'max_depth': [3, 5, 10,15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4], 
    'bootstrap': [True, False] 
}

auroc_scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr')


grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=auroc_scorer, verbose=1, n_jobs=-1)


grid_search.fit(X_train, y_train)



best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(y_pred)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")




classes = np.unique(y_test)
y_test_binarized = label_binarize(y_test, classes=classes)
y_pred_binarized = label_binarize(y_pred, classes=classes)

n_classes = len(classes) 
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_binarized[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])




for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for class {classes[i]}')
    plt.legend(loc="lower right")
    plt.show()

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))


mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])


mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure()
plt.plot(fpr["macro"], tpr["macro"], label='Macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Macro-average ROC Curve')
plt.legend(loc="lower right")
plt.show()



y_probs = best_model.predict_proba(X_test)
test_auroc = roc_auc_score(y_test, y_probs, multi_class='ovr')
print("Test Set AUROC Score for random forest: {:.3f}".format(test_auroc))

classes = np.unique(y_test)
y_test_binarized = label_binarize(y_test, classes=classes)
y_pred_binarized = label_binarize(y_pred, classes=classes)

n_classes = len(classes) 
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])




for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for class {classes[i]}')
    plt.legend(loc="lower right")
    plt.show()

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))


mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])


mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure()
plt.plot(fpr["macro"], tpr["macro"], label='Macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Macro-average ROC Curve')
plt.legend(loc="lower right")
plt.show()




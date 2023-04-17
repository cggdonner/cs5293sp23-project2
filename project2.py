# Catherine Donner
# CS 5293
# Project 2

# Import packages
import json
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
    
# Arguments to be put in terminal command using argparse
parser = argparse.ArgumentParser(description="Cluster recipes given ingredients")
parser.add_argument("--N", type=int, help="Number of closest cuisines")
parser.add_argument("--ingredient", type=str, nargs="+", help="Input ingredient")
args = parser.parse_args()
    
# Open and read in yummly.json
with open('yummly.json') as f:
    yummly = json.load(f)

# Preprocess yummly dataset and input ingredients in preparation for vectorization
all_ingredients = [', '.join(recipe['ingredients']) for recipe in yummly]
input_ingredients = ', '.join(args.ingredient)
# Using preprocessing
#all_cuisines = [recipe['cuisine'] for recipe in yummly]

# Split the data into testing and training sets
X_train, X_test = train_test_split(all_ingredients, test_size=0.2, random_state=42)

# Label encoding an training sets
#le = LabelEncoder()
#y_train = le.fit_transform(all_cuisines)

# Split the data into testing and training sets
#X_train, X_test, y_train, y_test = train_test_split(all_ingredients, y_train, test_size=0.2, random_state=42)

# Vectorize and normalize training set and input ingredients
cv = CountVectorizer()
X_train = cv.fit_transform(all_ingredients)
X_train_norm = normalize(X_train)

# Vectorize input ingredients
input_vec = cv.transform([input_ingredients])
input_vec_norm = normalize(input_vec)

# Train k-means model
kmeans = KMeans(n_clusters=len(set([recipe['cuisine'] for recipe in yummly])), n_init='auto')
#kmeans = KMeans(n_clusters=len(set(all_cuisines)), n_init='auto')
kmeans.fit(X_train_norm)

# Print silhouette score for verification of model performance
from sklearn.metrics import silhouette_score
silhouette = silhouette_score(X_train_norm, kmeans.labels_)
print("Silhouette score:", silhouette)

# Predict closest cuisine and n-closest cuisines
input_cluster = kmeans.predict(input_vec_norm)[0]
cluster_centers = kmeans.cluster_centers_
cosine_similarities = cosine_similarity(input_vec_norm, cluster_centers)[0]
top_cuisines = cosine_similarities.argsort()[::-1][:args.N]

# Prepare output in JSON format
output = {}
output['cuisine'] = yummly[kmeans.labels_[input_cluster]]['cuisine']
#output['cuisine'] = le.inverse_transform(np.atleast_1d(kmeans.labels_[input_cluster]))
output['score'] = cosine_similarities[input_cluster]
output['closest'] = []
for i in top_cuisines:
    closest = {}
    closest['id'] = yummly[kmeans.labels_[i]]['id']
    closest['score'] = cosine_similarities[i]
    output['closest'].append(closest)

# Print output
print(json.dumps(output, indent=2))

# Vectorize the yummly.json dataset using CountVectorizer
# This converts the ingredients to vectors
#cv = CountVectorizer()
#X = cv.fit_transform([', '.join(recipe['ingredients']) for recipe in yummly])
# Then scale the matrix data
#X_normalized = normalize(X)
#print(X_normalized)

# Split normalized data into train and test sets
#X_train, X_test = train_test_split(X_normalized, test_size=0.2, random_state=42)

# Fit KMeans model on the training data
#kmeans = KMeans(n_clusters=args.N, n_init=10, random_state=42).fit(X_train)

# Predict clusters for the test data
#test_clusters = kmeans.predict(X_test)

# Use the k-means model/search index to find the top-N closest cuisines
#closest_indices = []
#for i in range(args.N):
#    indices = np.argsort(cosine_similarity(X_normalized, kmeans.cluster_centers_)[::,i])[::-1]
#    closest_indices.append(indices[:args.N])

# Use cosine similarity to calculate similarity score between recipe and cluster center
#cosine = cosine_similarity(X_normalized, kmeans.cluster_centers_)

# Define the output to be returned in the terminal (will be in dictionary format)
#output = {}

# Then populate the output with closest cuisines
#for i in range(args.N):
#    n_closest_cuisine = [] # Initialize list to store strictly N-number of ids and scores
#    indices = np.argsort(cosine[:, i])[::-1][:args.N] # Get indices of top-N recipes based on cosine similarity to cluster center i
#    for j, index in enumerate(indices):
#        recipe = yummly[index]
#        n_closest_cuisine.append({
#            'id': recipe['id'],
#            'score': round(cosine[index, i], 2)
#        })
    # Sort N-closest recipes by descending score
#    n_closest_cuisine = sorted(n_closest_cuisine, key=lambda x: x['score'], reverse=True)[:args.N]

#    output[i] = {
#        'cuisine': yummly[indices[0]]['cuisine'], # use the recipe of the first index from indices (highest similarity score)
#        'score': round(cosine_similarity(kmeans.cluster_centers_[i].reshape(1, -1), X_normalized)[0][0], 2),
#        'closest': n_closest_cuisine
#    }

# Convert cluster index to cuisine name
#cuisine_names = kmeans.predict(X_normalized)
#cuisine_labels = kmeans.labels_
#cuisine_names_unique = np.unique(cuisine_names)
#cuisine_labels_unique = np.unique(cuisine_labels)
#cuisine_names_dict = {}
#for cuisine_name in cuisine_names_unique:
#    cuisine_names_dict[cuisine_name] = yummly[cuisine_labels_unique[cuisine_name]]['cuisine']

# Create final output dictionary
#final_output = {
#    'cuisine': cuisine_names_dict[cuisine_names[0]],
#    'score': round(cosine_similarity(kmeans.cluster_centers_[cuisine_names[0]].reshape(1, -1), X_normalized)[0][0], 2),
#    'closest': []
#}
#for i, closest_cuisine in enumerate(output[cuisine_names[0]]['closest']):
#    final_output['closest'].append({
#        'id': closest_cuisine['id'],
#        'score': closest_cuisine['score']
#    })

# Sort N-closest recipes by descending score
#n_closest_cuisine = sorted(n_closest_cuisine, key=lambda x: x['score'], reverse=True)[:args.N]
# Print the final output in json format
#print(json.dumps(final_output, indent=4))



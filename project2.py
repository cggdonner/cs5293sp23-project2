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
import os
import pickle
    
# Open and read in yummly.json
def read_json(filename):
    with open(filename) as f:
        return json.load(f)

# Preprocess yummly dataset in preparation for vectorization
def preprocess_dataset(dataset):
    all_ingredients = set()
    # Add any unique ingredient from the yummly dataset to all_ingredients and return the list of all unique ingredients
    for recipe in dataset:
        for ingredient in recipe['ingredients']:
            all_ingredients.add(ingredient)
    return list(all_ingredients)

# Preprocess input ingredients (args.ingredient) 
def preprocess_input(all_ingredients, ingredients):
    input_ingredients = {}
    # Add input ingredients to the dictionary as an intersection of all_ingredients and input_ingredients
    for ingredient in ingredients:
        if ingredient in all_ingredients:
            if ingredient in input_ingredients:
                input_ingredients[ingredient] += 1
            else:
                input_ingredients[ingredient] = 1
    # Return args.ingredients joined and how many of each ingredient put into the terminal
    return ', '.join([f'{k}' for k in input_ingredients.items()])

# Split the data into testing and training sets, set as 80% train/20% test
def train_split_data(X):
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    return X_train, X_test

# Vectorize and normalize training set and input ingredients using count vectorizer
def vectorize_data(X):
    cv = CountVectorizer() # Initialize count vectorizer
    X_vec = cv.fit_transform(X) # Fit data
    X_vec_norm = normalize(X_vec) # Then normalize
    return cv, X_vec_norm

# Train k-means model
def train_kmeans(X, num_clusters): 
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto') # n_init = 'auto' is used to prevent n_init warnings
    kmeans.fit(X) 
    return kmeans

# Predict closest cuisine and n-closest cuisines
def predict_cuisines(kmeans, input_vec_norm, n_closest_cuisines, dataset):
    input_cluster = kmeans.predict(input_vec_norm)[0] # Predict input cluster
    cluster_centers = kmeans.cluster_centers_ # Cluster centers of k-means model
    cosine_similarities = cosine_similarity(input_vec_norm, cluster_centers)[0] # Compute cosine similarities between the vectorized input ingredients and the cluster centers
    top_cuisines = cosine_similarities.argsort()[::-1][:n_closest_cuisines] # Sort n-closest cuisines in descending order
    
    # I also tried using Euclidean distance but JSON object not serializable
    #similarities = euclidean_distances(input_vec_norm, cluster_centers) # Use Euclidean distance instead of cosine similarity
    #scores = 1 / (1 + similarities) # Convert distances to similarities
    #top_cuisines = scores.argsort()[::-1][:n_closest_cuisines]

    # These print the cosine similarities and indices of the top cuisines
    #print(cosine_similarities)
    #print(top_cuisines)

    # Prepare output in JSON format
    output = {}
    output['cuisine'] = dataset[kmeans.labels_[input_cluster]]['cuisine'] # Returns closest cuisine using the k-means label of the input cluster
    output['score'] = cosine_similarities[input_cluster] # Computes the cosine similarity score of the input cluster
    #output['score'] = scores[0, input_cluster]
    output['closest'] = [] # Initialize n-closest cuisines list

    # Then append the n-closest cuisine ids and their scores to the closest list
    for i in top_cuisines:
        closest = {}
        closest['id'] = dataset[kmeans.labels_[i]]['id'] # added int()
        closest['score'] = cosine_similarities[i]
        #closest['score'] = scores[0, i]
        output['closest'].append(closest)

    # I tried creating a tracker to not duplicate cuisine ids, however the output returned fewer than the n-required closest cuisines (i.e. if args.N = 10, it would return 4 closest cuisines)
    # This issue could potentially be explained by overlaps in the k-means clusters where the input cluster still identifies with the same cuisine, however the cuisine can be in multiple different clusters with different scores, I tried separating the clusters more, however got syntax errors
    #selected_ids = set()
    #for i, cuisine_idx in enumerate(top_cuisines):
    #    if i >= n_closest_cuisines:
    #        break 

    #    cuisine_id = dataset[kmeans.labels_[cuisine_idx]]['id']
    #    if cuisine_id in selected_ids:
    #        continue

    return output

# Main function to call all previous functions and return output
def main(args):
    yummly = read_json('yummly.json')
    all_ingredients = preprocess_dataset(yummly)
    input_ingredients = preprocess_input(all_ingredients, args.ingredient)
    #print(input_ingredients) Verify input --ingredient are processed into project2.py
    X_train, X_test = train_split_data(all_ingredients)

    # I tried using tfidf vectorizer and HAC to separate the clusters more but had syntax issues that could not be resolved

    # Check if trained count vectorizer and k-means models already exist
    cv_filename = 'cv.pkl'
    kmeans_filename = 'kmeans.pkl'
    if os.path.exists(cv_filename) and os.path.exists(kmeans_filename):
        # Load trained models from file
        with open(cv_filename, 'rb') as f:
            cv = pickle.load(f)
        with open(kmeans_filename, 'rb') as f:
            kmeans = pickle.load(f)
    else:
        # Train new models and save them to file
        cv, X_train_norm = vectorize_data(X_train)
        with open(cv_filename, 'wb') as f:
            pickle.dump(cv, f)
        kmeans = train_kmeans(X_train_norm, len(set([recipe['cuisine'] for recipe in yummly])))
        with open(kmeans_filename, 'wb') as f:
            pickle.dump(kmeans, f)

    #cv, X_train_norm = vectorize_data(X_train)
    # Vectorize and normalize input_ingredients
    input_vec = cv.transform([input_ingredients])
    input_vec_norm = normalize(input_vec)
    #kmeans = train_kmeans(X_train_norm, len(set([recipe['cuisine'] for recipe in yummly])))
    output = predict_cuisines(kmeans, input_vec_norm, args.N, yummly)
    # Print output
    print(json.dumps(output, indent=2))

# Argparser to include terminal arguments into main, --ingredient can be called one or multiple times in the terminal
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cluster recipes given ingredients")
    parser.add_argument("--N", type=int, help="Number of closest cuisines")
    parser.add_argument("--ingredient", type=str, action='append', help="Input ingredient")
    args = parser.parse_args()
    main(args)

# End of code



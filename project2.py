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
    
# Open and read in yummly.json
def read_json(filename):
    with open(filename) as f:
        return json.load(f)

# Preprocess yummly dataset and input ingredients in preparation for vectorization
def preprocess_dataset(dataset):
    all_ingredients = set()
    for recipe in dataset:
        for ingredient in recipe['ingredients']:
            all_ingredients.add(ingredient)
    return list(all_ingredients)

def preprocess_input(all_ingredients, ingredients):
    input_ingredients = {}
    for ingredient in ingredients:
        if ingredient in all_ingredients:
            if ingredient in input_ingredients:
                input_ingredients[ingredient] += 1
            else:
                input_ingredients[ingredient] = 1
    return ', '.join([f'{k} ({v})' for k, v in input_ingredients.items()])

# Split the data into testing and training sets
def train_split_data(X):
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    return X_train, X_test

# Vectorize and normalize training set and input ingredients
def vectorize_data(X):
    cv = CountVectorizer()
    X_vec = cv.fit_transform(X)
    X_vec_norm = normalize(X_vec)
    return cv, X_vec_norm

# Train k-means model
def train_kmeans(X, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
    kmeans.fit(X)
    return kmeans

# Predict closest cuisine and n-closest cuisines
def predict_cuisines(kmeans, input_vec_norm, n_closest_cuisines, dataset):
    input_cluster = kmeans.predict(input_vec_norm)[0]
    cluster_centers = kmeans.cluster_centers_
    cosine_similarities = cosine_similarity(input_vec_norm, cluster_centers)[0]
    top_cuisines = cosine_similarities.argsort()[::-1][:n_closest_cuisines]

    # Prepare output in JSON format
    output = {}
    output['cuisine'] = dataset[kmeans.labels_[input_cluster]]['cuisine']
    output['score'] = cosine_similarities[input_cluster]
    output['closest'] = []
    for i in top_cuisines:
        closest = {}
        closest['id'] = dataset[kmeans.labels_[i]]['id']
        closest['score'] = cosine_similarities[i]
        output['closest'].append(closest)
    return output

# Main function to call all previous functions and return output
def main(args):
    yummly = read_json('yummly.json')
    all_ingredients = preprocess_dataset(yummly)
    input_ingredients = preprocess_input(all_ingredients, args.ingredient)
    print(all_ingredients)
    print(input_ingredients)
    X_train, X_test = train_split_data(all_ingredients)
    cv, X_train_norm = vectorize_data(X_train)
    input_vec = cv.transform([input_ingredients])
    input_vec_norm = normalize(input_vec)
    kmeans = train_kmeans(X_train_norm, len(set([recipe['cuisine'] for recipe in yummly])))
    output = predict_cuisines(kmeans, input_vec_norm, args.N, yummly)
    # Print output
    print(json.dumps(output, indent=2))

# Argparser to include terminal arguments into main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cluster recipes given ingredients")
    parser.add_argument("--N", type=int, help="Number of closest cuisines")
    parser.add_argument("--ingredient", type=str, action='append', help="Input ingredient")
    args = parser.parse_args()
    main(args)



# Catherine Donner
# CS 5293
# Project 2

# Import packages
import json
import argparse
import numpy as np
from sklearn.cluster import KMeans # Originally used Kmeans and count vectorizer
from sklearn.metrics.pairwise import cosine_similarity # To calculate scores
from sklearn.model_selection import train_test_split 
import os
import pickle # To save tfidf and knn models
import pandas as pd # For dataframe storage
from sklearn.feature_extraction.text import TfidfVectorizer # For vectorization
from sklearn.neighbors import KNeighborsClassifier # For training

# Open and read in yummly.json
def read_json(filename):
    with open(filename) as f:
        return json.load(f)

# Convert the json file to a pandas dataframe for easier training
def convert_to_df(json):
    df = pd.DataFrame(json)
    return df

# Preprocess yummly dataset in preparation for vectorization
def preprocess_dataset(df):
    # Join all ingredients in the dataframe together
    all_ingredients = df['ingredients']
    ingredients = list(map(' '.join, all_ingredients))
    return ingredients

# Preprocess input ingredients (args.ingredient) 
def preprocess_input(all_ingredients, ingredients):
    # Add input ingredients to the dataframe ingredients for finding relationship between cuisines and input ingredients
    input_ingredients = all_ingredients.append(' '.join(ingredients))
    return input_ingredients

# Split the dataframe into testing and training sets, set as 80% train/20% test
def train_split_data(X):
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    return X_train, X_test

# Vectorize training set using TfidfVectorizer
def vectorize_data(X):
    tfidf = TfidfVectorizer() # Initialize TfidfVectorizer
    X_vec = tfidf.fit_transform(X)
    return tfidf, X_vec

# Train k-nearest neighbors model using the vectorized training set and the cuisines in the training set (Y), with num_neighbors being args.N cuisines
def train_knn(X_vec, Y, num_neighbors):
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    knn.fit(X_vec, Y)
    return knn

# Predict closest cuisine and n_closest cuisines
# This takes in the knn model, the vectorized training set, the vectorized input, args.N, and the dataframe, predicts the cuisines and returns the output
def predict_cuisines(knn, X_vec, input_vec, n_closest_cuisines, dataset):
    predictions = list(knn.predict(input_vec)) # Makes the predictions using the vectorized input
    predicted_score = round(np.max(knn.predict_proba(input_vec)), 2) # Returns highest cuisine score rounded to two places
    cosine_similarities = cosine_similarity(input_vec, X_vec).ravel() # Finds cosine similarity between input vector and vectorized training set, need ravel to put scores in 1D array
    top_cuisines = cosine_similarities.argsort()[::-1][:n_closest_cuisines]
    # Now set up output format
    output = {}
    output['cuisine'] = predictions[0] # First element of predictions
    output['score'] = predicted_score # Highest cuisine score
    output['closest'] = []
    # For id and score in n_closest cuisines, append them
    for i in top_cuisines:
        closest = {}
        closest['id'] = int(dataset['id'][i]) 
        closest['score'] = round(cosine_similarities[i], 2)
        output['closest'].append(closest)

    return output

# Main function to call all previous functions and return output
def main(args):
    # Read in json file
    yummly = read_json('yummly.json')
    # Convert the json file to the pandas dataframe
    df = convert_to_df(yummly)
    # Split the dataframe into training and testing sets
    X_train, X_test = train_split_data(df)
    # Join all ingredients in the training set to be vectorized
    all_ingredients = preprocess_dataset(X_train) 
    # Join input ingredients with all_ingredients
    input_ingredients = preprocess_input(all_ingredients, args.ingredient)

    # Check if trained tfidf vectorizer and knn models already exist
    tfidf_filename = 'tfidf.pkl'
    knn_filename = 'knn.pkl'
    if os.path.exists(tfidf_filename) and os.path.exists(knn_filename):
        # Load trained models from file
        with open(tfidf_filename, 'rb') as f:
            tfidf = pickle.load(f)
        with open(knn_filename, 'rb') as f:
            knn = pickle.load(f)
        X_vec = tfidf.transform(all_ingredients[:-1])
    else:
        # Train new models and save them to file
        tfidf, X_vec = vectorize_data(all_ingredients[:-1])
        with open(tfidf_filename, 'wb') as f:
            pickle.dump(tfidf, f)
        knn = train_knn(X_vec, X_train['cuisine'], args.N) # Train knn model with vectorized X_train ingredients and the cuisine column in X_train 
        with open(knn_filename, 'wb') as f:
            pickle.dump(knn, f)
    
    # Vectorize input ingredients (last element of all_ingredients list)
    input_vec = tfidf.transform([all_ingredients[-1]]) # [all_ingredients[-1]] are the input ingredients located at the end of all_ingredients
    # Predict cuisines
    output = predict_cuisines(knn, X_vec, input_vec, args.N, df)
    # Print output in json format
    print(json.dumps(output, indent=2))

# Argparser to include terminal arguments into main, --ingredient can be called one or multiple times in the terminal
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cluster recipes given ingredients")
    parser.add_argument("--N", type=int, help="Number of closest cuisines")
    parser.add_argument("--ingredient", type=str, action='append', help="Input ingredient")
    args = parser.parse_args()
    main(args)

    




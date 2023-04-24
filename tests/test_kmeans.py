# Import project2.py
from project2 import read_json
from project2 import preprocess_dataset
from project2 import preprocess_input
from project2 import train_split_data
from project2 import vectorize_data
from project2 import train_kmeans
import os
import pickle
import numpy as np

# Function to test the k-means model setup (train_kmeans method)
def test_kmeans():
    yummly = read_json('yummly.json')
    all_ingredients = preprocess_dataset(yummly)
    sample = ['Oscar Mayer Bacon', 'Kraft Grated Parmesan Cheese', 'Philadelphia Cream Cheese', 'corn tortillas']
    input_ingredients = preprocess_input(all_ingredients, sample)
    X_train, X_test = train_split_data(all_ingredients)

    # Check if trained models already exist
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
        num_clusters = len(set([recipe['cuisine'] for recipe in yummly]))
        kmeans = train_kmeans(X_train_norm, num_clusters)
        # Assert number of k-means labels equals the number of clusters
        assert len(set(kmeans.labels_)) == num_clusters # Number of kmeans labels equal to number of clusters
        with open(kmeans_filename, 'wb') as f:
            pickle.dump(kmeans, f)
        
    # More assertions to make sure that the number of k-means clusters are finite
    assert not np.any(np.isnan(kmeans.cluster_centers_)) # No NaN cluster centers
    assert not np.any(np.isinf(kmeans.cluster_centers_)) # No infinite cluster centers
    

# Import project2.py
from project2 import read_json
from project2 import preprocess_dataset
from project2 import preprocess_input
from project2 import train_split_data
from project2 import vectorize_data
from project2 import train_kmeans
from project2 import predict_cuisines
import os
import pickle
from sklearn.preprocessing import normalize
import json

# Function to test output
def test_output():
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
        kmeans = train_kmeans(X_train_norm, len(set([recipe['cuisine'] for recipe in yummly])))
        with open(kmeans_filename, 'wb') as f:
            pickle.dump(kmeans, f)

    input_vec = cv.transform([input_ingredients])
    input_vec_norm = normalize(input_vec)
    #kmeans = train_kmeans(X_train_norm, len(set([recipe['cuisine'] for recipe in yummly])))
    output = predict_cuisines(kmeans, input_vec_norm, 3, yummly) # Set to 3 clusters
    # Print output
    print(json.dumps(output, indent=2))

    # Assert that the json output (no indentation implemented) is exact
    assert json.dumps(output) == '''{"cuisine": "southern_us", "score": 0.5746755488386242, "closest": [{"id": 25693, "score": 0.5746755488386242}, {"id": 25693, "score": 0.2441982701357346}, {"id": 25693, "score": 0.15173148570319275}]}'''
        

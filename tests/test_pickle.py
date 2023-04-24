# Import packages
from project2 import read_json
from project2 import preprocess_dataset
from project2 import preprocess_input
from project2 import train_split_data
from project2 import vectorize_data
from project2 import train_kmeans
import os
import pickle

# Function to test the pickling of the k-means and count vectorizer models
def test_pickle():
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
        # Assertion to make sure that the model is saved (if cv is not None and kmeans is not None)
        assert cv is not None and kmeans is not None
        print(f"The path exists, so the model is saved already.")
    else:
        # Train new models and save them to file
        cv, X_train_norm = vectorize_data(X_train)
        with open(cv_filename, 'wb') as f:
            pickle.dump(cv, f)
        kmeans = train_kmeans(X_train_norm, len(set([recipe['cuisine'] for recipe in yummly])))
        with open(kmeans_filename, 'wb') as f:
            pickle.dump(kmeans, f)
        print(f"The path does not exist, therefore the model had to be created.")


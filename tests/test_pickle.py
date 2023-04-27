# Import packages
from project2 import read_json
from project2 import convert_to_df
from project2 import preprocess_dataset
from project2 import preprocess_input
from project2 import train_split_data
from project2 import vectorize_data
from project2 import train_knn
import os
import pickle

# Function to test the pickling of the knn and tfidf vectorizer models
def test_pickle():
    yummly = read_json('yummly.json')
    df = convert_to_df(yummly)
    X_train, X_test = train_split_data(df)
    all_ingredients = preprocess_dataset(df)
    sample = ['Oscar Mayer Bacon', 'Kraft Grated Parmesan Cheese', 'Philadelphia Cream Cheese', 'corn tortillas']
    input_ingredients = preprocess_input(all_ingredients, sample)

    # Check if trained models already exist
    tfidf_filename = 'tfidf.pkl'
    knn_filename = 'knn.pkl'
    if os.path.exists(tfidf_filename) and os.path.exists(knn_filename):
        # Load trained models from file
        with open(tfidf_filename, 'rb') as f:
            tfidf = pickle.load(f)
        with open(knn_filename, 'rb') as f:
            knn = pickle.load(f)
        # Assert that models have been saved
        assert tfidf is not None
        assert knn is not None
    else:
        # Train new models and save them to file
        tfidf, X_vec = vectorize_data(all_ingredients[:-1])
        with open(tfidf_filename, 'wb') as f:            pickle.dump(tfidf, f)
        knn = train_knn(X_vec, X_train['cuisine'], 3) # Set 3 n-closest clusters
        with open(knn_filename, 'wb') as f:
            pickle.dump(knn, f)

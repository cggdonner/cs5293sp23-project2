# Import project2.py
from project2 import read_json
from project2 import convert_to_df
from project2 import preprocess_dataset
from project2 import preprocess_input
from project2 import train_split_data
from project2 import vectorize_data
from project2 import train_knn
from project2 import predict_cuisines
import os
import pickle
import json

# Function to test output
def test_output():
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
        X_vec = tfidf.transform(all_ingredients[:-1])
    else:
        # Train new models and save them to file
        tfidf, X_vec = vectorize_data(all_ingredients[:-1])
        with open(tfidf_filename, 'wb') as f:
            pickle.dump(tfidf, f)
        knn = train_knn(X_vec, X_train['cuisine'], 3) # Set 3 n-closest neighbors
        with open(knn_filename, 'wb') as f:
            pickle.dump(knn, f)

    input_vec = tfidf.transform([all_ingredients[-1]])
    output = predict_cuisines(knn, X_vec, input_vec, 3, df) # Again set to 3 nearest cuisines

    # Assert that the json output (no indentation implemented) is exact
    assert json.dumps(output) == '''{"cuisine": "italian", "score": 0.6, "closest": [{"id": 18528, "score": 0.91}, {"id": 21444, "score": 0.83}, {"id": 37236, "score": 0.73}]}''' 

# Import project2.py
from project2 import read_json
from project2 import preprocess_dataset
from project2 import preprocess_input
from project2 import train_split_data
from project2 import vectorize_data
from project2 import train_kmeans

def test_kmeans():
    yummly = read_json('yummly.json')
    all_ingredients = preprocess_dataset(yummly)
    sample = ['Oscar Mayer Bacon', 'Kraft Grated Parmesan Cheese', 'Philadelphia Cream Cheese', 'corn tortillas']
    input_ingredients = preprocess_input(all_ingredients, sample)
    X_train, X_test = train_split_data(all_ingredients)
    cv, X_train_norm = vectorize_data(X_train)
    input_vec = cv_transform([input_ingredients])
    input_vec_norm = normalize(input_vec)
    kmeans = train_kmeans(X_train_norm), len(set([recipe['cuisine'] for recipe in yummly])))
    # Assertions
    assert not np.any(np.isnan(kmeans.cluster_centers_)) # No NaN cluster centers
    assert not np.any(np.isinf(kmeans.cluster_centers_)) # No infinite cluster centers
    assert len(set(kmeans.labels_)) == num_clusters # Number of kmeans labels equal to number of clusters

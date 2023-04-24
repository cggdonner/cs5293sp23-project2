# Import project2.py
from project2 import read_json
from project2 import preprocess_dataset
from project2 import preprocess_input
from project2 import train_split_data
import pytest

# Function to test and call train_split_data()
def test_split():
    yummly = read_json('yummly.json')
    all_ingredients = preprocess_dataset(yummly)
    sample = ['Oscar Mayer Bacon', 'Kraft Grated Parmesan Cheese', 'Philadelphia Cream Cheese', 'corn tortillas']
    input_ingredients = preprocess_input(all_ingredients, sample)
    X_train, X_test = train_split_data(all_ingredients)
    # Assertions (make sure it is an 80%/20% split)
    assert len(X_train) == pytest.approx(0.8*len(all_ingredients), abs=1)
    assert len(X_test) == pytest.approx(0.2*len(all_ingredients), abs=1)

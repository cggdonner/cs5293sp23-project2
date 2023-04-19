# Import project2.py
from project2 import read_json
from project2 import preprocess_dataset
from project2 import preprocess_input

# Function to test and call read_json(), preprocess_dataset(), and preprocess_input()
def test_preprocess():
    yummly = read_json('yummly.json')
    all_ingredients = preprocess_dataset(yummly)
    sample = ['Oscar Mayer Bacon', 'Kraft Grated Parmesan Cheese', 'Philadelphia Cream Cheese', 'corn tortillas']
    input_ingredients = preprocess_input(all_ingredients, sample)
    # Assertions
    assert len(all_ingredients) == 1000
    assert len(input_ingredients) == len(sample)

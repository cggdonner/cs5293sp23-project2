# Import project2.py
from project2 import read_json
from project2 import preprocess_dataset
from project2 import preprocess_input

# Function to test and call preprocess_dataset() and preprocess_input()
def test_preprocess():
    yummly = read_json('yummly.json')
    all_ingredients = preprocess_dataset(yummly)
    # Instead of args.ingredient, I will be using the ingredients in sample list as the input ingredients
    sample = ['Oscar Mayer Bacon', 'Kraft Grated Parmesan Cheese', 'Philadelphia Cream Cheese', 'corn tortillas']
    input_ingredients = preprocess_input(all_ingredients, sample)
    # Assertions
    assert len(all_ingredients) == 6714
    assert len(input_ingredients.split(', ')) == 8 # Length 8 because of ingredients and counts

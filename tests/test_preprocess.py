# Import project2.py
from project2 import read_json
from project2 import convert_to_df
from project2 import train_split_data
from project2 import preprocess_dataset
from project2 import preprocess_input

# Function to test and call preprocess_dataset() and preprocess_input()
def test_preprocess():
    yummly = read_json('yummly.json')
    df = convert_to_df(yummly)
    X_train, X_test = train_split_data(df)
    all_ingredients = preprocess_dataset(X_train)
    # Instead of args.ingredient, I will be using the ingredients in sample list as the input ingredients
    sample = ['Oscar Mayer Bacon', 'Kraft Grated Parmesan Cheese', 'Philadelphia Cream Cheese', 'corn tortillas']
    input_ingredients = preprocess_input(all_ingredients, sample)
    concat_sample = ' '.join(sample).replace(',', ' ')
    # Assertions
    assert len(all_ingredients) == 31820
    assert input_ingredients is None
    assert len(all_ingredients[-1]) == len(concat_sample)

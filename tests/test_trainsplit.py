# Open and read in yummly.json
with open('yummly.json') as f:
    yummly = json.load(f)

# Preprocess yummly dataset and input ingredients in preparation for vectorization
all_ingredients = [', '.join(recipe['ingredients']) for recipe in yummly]
input_ingredients = ['garlic cloves', 'pepper', 'feta cheese crumbles']

# Split the data into testing and training sets
X_train, X_test = train_test_split(all_ingredients, test_size=0.2, random_state=42)

assert len(X_train) == 800
assert len(X_test) == 200

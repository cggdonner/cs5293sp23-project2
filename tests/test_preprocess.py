# Open and read in yummly.json
with open('yummly.json') as f:
    yummly = json.load(f)

# Preprocess yummly dataset and input ingredients in preparation for vectorization
all_ingredients = [', '.join(recipe['ingredients']) for recipe in yummly]
input_ingredients = ['garlic cloves', 'pepper', 'feta cheese crumbles']

assert len(all_ingredients) == 1000

import json

# Open yummly dataset
with open('yummly.json') as f:
    yummly = json.load(f)

assert len(yummly) == 100

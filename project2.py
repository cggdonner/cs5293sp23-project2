# Catherine Donner
# CS 5293
# Project 2

# Import packages
import json
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
    
# Arguments to be put in terminal command using argparse
parser = argparse.ArgumentParser(description="Cluster recipes given ingredients")
parser.add_argument("--N", type=int, help="Number of closest cuisines")
parser.add_argument("--ingredient", type=str, nargs="+", help="Input ingredient")
args = parser.parse_args()
    
# Open and read in yummly.json
with open('yummly.json') as f:
    data = json.load(f)

# First filter the yummly data to only include recipes that contain the input ingredients
filtered_data = [] # Initialize list object for storage of filtered data
for recipe in data:
    if any(i in recipe['ingredients'] for i in args.ingredient):
        filtered_data.append(recipe) # Append any recipes/cuisines that contain the ingredients listed in the terminal arguments

# Use one hot encoding to create matrix to be populated with binary values based on if the ingredient is present in the filtered data
# First take all of the unique ingredients listed in the filtered data
all_ingredients = set()
for recipe in filtered_data:
    all_ingredients.update(recipe['ingredients'])

# Sort the list of unique ingredients
unique_ingredients = sorted(list(all_ingredients))

# Create 2D array and set all values to 0
X = np.zeros((len(filtered_data), len(unique_ingredients)))
for i, recipe in enumerate(filtered_data):
    for j, ingredient in enumerate(unique_ingredients):
        if ingredient in recipe['ingredients']:
            X[i,j] = 1 # set any entries in the matrix to 1 if the ingredient is present in the recipe

# Initialize k-means clustering algorithm with N clusters
kmeans = KMeans(n_clusters=args.N, n_init=10) # Had to add n_init=10 to suppress warning
kmeans.fit(X)

# Use the k-means model/search index to find the top-N closest cuisines
closest_indices = []
for i in range(args.N):
    indices = np.argsort(np.linalg.norm(X - kmeans.cluster_centers_[i], axis=1))
    closest_indices.append(indices[:args.N])

# Define the output to be returned in the terminal (will be in dictionary format)
output = {
        'cuisine': recipe['cuisine'],
        'score': 1 / (1 + abs(kmeans.score(X))),
        'closest': []
        }

# Use cosine similarity to calculate similarity score between recipe and cluster center
cosine = cosine_similarity(X, kmeans.cluster_centers_)

# Then populate the 'closest' attribute of the output with N-closest ids and scores
for i in range(args.N):
    n_closest_cuisine = [] # Initialize list to store strictly N-number of ids and scores
    indices = np.argsort(cosine[:, i])[::-1][:args.N] # Get indices of top-N recipes based on cosine similarity to cluster center i
    for j, index in enumerate(indices):
        recipe = filtered_data[index]
        n_closest_cuisine.append({
            'id': recipe['id'],
            'score': cosine[index, i]
        })

# Sort N-closest recipes by descending score 
n_closest_cuisine = sorted(n_closest_cuisine, key=lambda x: x['score'], reverse=True)[:args.N]
output['closest'] += n_closest_cuisine

# Print the final output in json format
print(json.dumps(output, indent=4))



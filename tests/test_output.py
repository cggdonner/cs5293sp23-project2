# Open and read in yummly.json
with open('yummly.json') as f:
    yummly = json.load(f)

# Preprocess yummly dataset and input ingredients in preparation for vectorization
all_ingredients = [', '.join(recipe['ingredients']) for recipe in yummly]
input_ingredients = ['garlic cloves', 'pepper', 'feta cheese crumbles']

# Split the data into testing and training sets
X_train, X_test = train_test_split(all_ingredients, test_size=0.2, random_state=42)

# Vectorize and normalize training set and input ingredients
cv = CountVectorizer()
X_train = cv.fit_transform(all_ingredients)
X_train_norm = normalize(X_train)

# Vectorize input ingredients
input_vec = cv.transform([input_ingredients])
input_vec_norm = normalize(input_vec)

# Train k-means model
kmeans = KMeans(n_clusters=len(set([recipe['cuisine'] for recipe in yummly])), n_init='auto')
kmeans.fit(X_train_norm)

# Predict closest cuisine and n-closest cuisines
input_cluster = kmeans.predict(input_vec_norm)[0]
cluster_centers = kmeans.cluster_centers_
cosine_similarities = cosine_similarity(input_vec_norm, cluster_centers)[0]
top_cuisines = cosine_similarities.argsort()[::-1][:3]

# Prepare output in JSON format
output = {}
output['cuisine'] = yummly[kmeans.labels_[input_cluster]]['cuisine']
output['score'] = cosine_similarities[input_cluster]
output['closest'] = []
for i in top_cuisines:
    closest = {}
    closest['id'] = yummly[kmeans.labels_[i]]['id']
    closest['score'] = cosine_similarities[i]
    output['closest'].append(closest)

# Assertions
assert output['cuisine'] == 'japanese'
assert output['score'] == 0.121526
assert output['closest'] == [{'id': 55072, 'score': 0.102345, 'id': 4402, 'score': 0.09348483, 'id': 490, 'score': 0.083384}]

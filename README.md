# cs5293sp23-project2
## Authored by Catherine Donner (cggdonner)
Repo used for Text Analytics Project 2 workspace

## Project Description
This project is designed to train the provided food data in the yummly.json dataset, asks the user to input whichever ingredients they want to predict a cuisine for as command line arguments, uses the model to predict the type of cuisine and tell the user, and finds the N-closest cuisines (as args.N in the command line) and returns the IDs of those cuisines to the user.

## How to Install
You can install my Project 2 using the command: git clone https://github.com/cggdonner/cs5293sp23-project2.git.

You will have to sign into the repository with your credentials as a collaborator. After cloning the GitHub repo into your terminal, you can access it using the command cd cs5293sp23-project2/. To view all files and directories in the terminal, use the command tree .

## How to Run
In the terminal, you use the following command to run project2.py in the pip environment:

pipenv run python project2.py --N 5 --ingredient paprika --ingredient "romaine lettuce" --ingredient "feta cheese crumbles"

where the argument following the --ingredient flag can be any ingredient that you wish to input in the terminal and it can also be duplicated as many times as possible.

To run test files in the terminal, use the command:

pipenv run python -m pytest

## Methods in project2.py
read_json() - This takes in the parameter filename and returns json.load(f) where f is the opened filename. This method reads in the yummly.json dataset and returns the read-in dataset.

preprocess_dataset() - This takes in the parameter dataset and returns list(all_ingredients). This method adds all unique ingredients from the yummly dataset to a list object called all_ingredients and returns that list.

preprocess_input() - This takes in the parameters all_ingredients and ingredients. It takes the intersection of all_ingredients and the input ingredients and returns the input ingredients as well as each count of each ingredient that is put into the terminal (returns return ', '.join([f'{k} ({v})' for k, v in input_ingredients.items()])).

train_split_data() - This takes in the parameter X and returns X_train and X_test. This splits all_ingredients into a training set of 80% size and a testing set of 20% size.

vectorize_data() - This takes in the parameter X and returns cv and X_vec_norm. This initializes and fits the CountVectorizer model to the X_train data and normalizes the X_train data, returning the CV model and the normalized data.

train_kmeans() - This takes in the parameters X and num_clusters. This method fits the k-means clustering model with the normalized data and the number of cuisines in the dataset as the number of clusters. It returns the model variable kmeans.

predict_cuisines() - This takes in the parameters kmeans, input_vec_norm, n_closest_cuisines, and dataset, and returns output. This computes the input cluster, cluster centers, cosine similarities between the input cluster and cluster centers (to be used as scores) and the n-top cuisines sorted in descending order. Then the output is created in JSON format with the n-closest cuisines appended to the 'closest' list.

main() - The main function takes args (args.N and args.ingredient) and calls all previous functions, printing the output. The train_kmeans() and vectorize_data() methods are called in an if-conditional using the pickle package if the k-means model has not already been saved. Then input_ingredients is vectorized and normalized with the same CountVectorizer model and put into the predict_cuisines() method.

## Tests
For testing project2.py, I created these tests:

test_read() - 

test_preprocess() - 

test_train_split() - 

test_cv() - 

test_kmeans() - 

test_pickle() - 

test_output() - 

## Bugs and Assumptions
When putting the --ingredient flags into the terminal, one-word ingredients do not have to have double quotes around them, however for multiple-word ingredients double quotes are required.
When putting --ingredient flags into the terminal, the program will only recognize the ingredients exactly as listed in the yummly.json file. If you use a shortened version of the ingredient, the program will not count it (i.e. put "feta cheese crumbles" instead of "feta cheese").
**MORE BUGS TO LIST**



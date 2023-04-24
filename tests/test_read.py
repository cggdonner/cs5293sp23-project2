# Import project2.py
from project2 import read_json

# Function to call and test read_json()
def test_read():
    yummly = read_json('yummly.json')
    # Assert that the length of the dataset is at least 1 (then dataset has been imported)
    assert len(yummly) >= 1

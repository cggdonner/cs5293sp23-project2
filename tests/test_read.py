# Import project2.py
from project2 import read_json

# Function to call and test read_json()
def test_read():
    yummly = read_json('yummly.json')
    # Assertion
    assert len(yummly) >= 1

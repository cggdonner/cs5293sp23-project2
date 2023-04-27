# Import project2.py
from project2 import read_json
from project2 import convert_to_df

# Function to test convert_to_df()
def test_df():
    yummly = read_json('yummly.json')
    df = convert_to_df(yummly)
    # Assertion
    assert df.shape == (39774, 3)

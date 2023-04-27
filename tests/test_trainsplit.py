# Import project2.py
from project2 import read_json
from project2 import convert_to_df
from project2 import train_split_data
import pytest

# Function to test and call train_split_data()
def test_split():
    yummly = read_json('yummly.json')
    df = convert_to_df(yummly)
    X_train, X_test = train_split_data(df)
    # Assertions (make sure it is an 80%/20% split)
    assert len(X_train) == pytest.approx(0.8*len(df), abs=1)
    assert len(X_test) == pytest.approx(0.2*len(df), abs=1)

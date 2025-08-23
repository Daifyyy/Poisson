import pickle
from pathlib import Path

def dump(value, filename):
    with open(Path(filename), 'wb') as f:
        pickle.dump(value, f)

def load(filename):
    with open(Path(filename), 'rb') as f:
        return pickle.load(f)

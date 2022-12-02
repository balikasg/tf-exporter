import json


def load_json(path):
    """Loads the contents of a json file."""
    with open(path, "r") as infile:
        return json.load(infile)

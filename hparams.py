import json

class HParams():
    """
    Load hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.lr)
    params.lr = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        """Initialization."""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        """Save parameters to json file."""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Load parameters from json file."""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Give dict-like access to Params instance by `params.dict['learning_rate']."""
        return self.__dict__
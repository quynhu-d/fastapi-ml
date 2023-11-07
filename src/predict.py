import pickle
import pandas as pd
from src.data import Item

#TODO: predict for several
def predict(item: Item, model_path: str = './models/svr.pkl'):
    """
    Predict grade for one student.

    Parameters:
        item (Item): sample features
        model_path (str): path to fitted model
    Returns:
        prediction (float): grade prediction
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(model)
    features_in = model.feature_names_in_
    print(features_in)
    test_sample = pd.Series(item.dict())[features_in].to_frame().T
    print(test_sample)
    prediction = model.predict(test_sample)[0]
    return prediction
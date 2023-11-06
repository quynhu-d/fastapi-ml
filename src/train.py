from model import BaseModel
def train_model(model: BaseModel, data: tuple):
    x, y = data
    model.fit(x, y)
    return model
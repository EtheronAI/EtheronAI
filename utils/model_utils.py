from sklearn.linear_model import SGDRegressor

def initialize_model(data=None, target=None):
    model = SGDRegressor()
    if data is not None and target is not None:
        model.partial_fit(data, target)
    return model

def update_model(model, data, target):
    model.partial_fit(data, target)
    return model

def predict_model(model, data):
    return model.predict(data)
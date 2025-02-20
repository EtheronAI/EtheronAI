from flask import Blueprint, request, jsonify
import numpy as np
from sklearn.linear_model import SGDRegressor
from utils.model_utils import initialize_model, update_model, predict_model

# Create blueprint
adaptive_learning_bp = Blueprint('adaptive_learning', __name__)

# Initialize model
model = initialize_model()

@adaptive_learning_bp.route('/init', methods=['POST'])
def init_model():
    """
    Initialize Model API
    ---
    parameters:
      - name: data
        in: body
        type: array
        required: true
        description: Initial training data
      - name: target
        in: body
        type: array
        required: true
        description: Initial target values
    responses:
      200:
        description: Model initialized successfully
        schema:
          type: object
          properties:
            message:
              type: string
    """
    data = request.json['data']
    target = request.json['target']
    model = initialize_model(data, target)
    return jsonify({'message': 'Model initialized successfully'}), 200

@adaptive_learning_bp.route('/update', methods=['POST'])
def update_model():
    """
    Update Model API
    ---
    parameters:
      - name: data
        in: body
        type: array
        required: true
        description: New data
      - name: target
        in: body
        type: array
        required: true
        description: New target values
    responses:
      200:
        description: Model updated successfully
        schema:
          type: object
          properties:
            message:
              type: string
    """
    data = request.json['data']
    target = request.json['target']
    model = update_model(model, data, target)
    return jsonify({'message': 'Model updated successfully'}), 200

@adaptive_learning_bp.route('/predict', methods=['POST'])
def predict():
    """
    Predict API
    ---
    parameters:
      - name: data
        in: body
        type: array
        required: true
        description: Input data
    responses:
      200:
        description: Prediction result
        schema:
          type: object
          properties:
            prediction:
              type: array
    """
    data = request.json['data']
    prediction = predict_model(model, data)
    return jsonify({'prediction': prediction.tolist()}), 200
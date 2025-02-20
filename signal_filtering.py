from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from utils.data_processing import preprocess_data, postprocess_data

# Create blueprint
signal_filtering_bp = Blueprint('signal_filtering', __name__)

# Load pre-trained model
model = tf.keras.models.load_model('signal_filter_model.h5')

# Initialize scaler
scaler = StandardScaler()

@signal_filtering_bp.route('/filter', methods=['POST'])
def filter_signal():
    """
    Signal Filtering API
    ---
    parameters:
      - name: data
        in: body
        type: array
        required: true
        description: Time series data (signal + noise)
    responses:
      200:
        description: Filtered signal data
        schema:
          type: object
          properties:
            filtered_signal:
              type: array
              description: Filtered signal
    """
    try:
        data = request.json['data']
        data = np.array(data).reshape(-1, 1)

        # Preprocess data
        data_scaled = preprocess_data(data, scaler)

        # Predict using the model
        filtered_signal = model.predict(data_scaled)

        # Postprocess data
        filtered_signal = postprocess_data(filtered_signal, scaler)

        return jsonify({'filtered_signal': filtered_signal.flatten().tolist()}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(data, scaler):
    data = np.array(data).reshape(-1, 1)
    return scaler.fit_transform(data)

def postprocess_data(data, scaler):
    return scaler.inverse_transform(data)

def preprocess_text(text, language):
    if language == 'zh':
        return f"用中文生成内容：{text}"
    elif language == 'es':
        return f"Generar contenido en español: {text}"
    return text

def postprocess_text(text, language):
    return text
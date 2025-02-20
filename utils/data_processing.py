import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(data, scaler):
    data = np.array(data).reshape(-1, 1)
    return scaler.fit_transform(data)

def postprocess_data(data, scaler):
    return scaler.inverse_transform(data)

def preprocess_text(text, language):
    """
    Preprocess text based on language.
    """
    if language == 'zh':
        return f"用中文生成内容：{text}"
    elif language == 'es':
        return f"Generar contenido en español: {text}"
    elif language == 'fr':
        return f"Générer du contenu en français: {text}"
    elif language == 'de':
        return f"Generiere Inhalte auf Deutsch: {text}"
    elif language == 'ja':
        return f"日本語でコンテンツを生成する：{text}"
    elif language == 'ko':
        return f"한국어로 콘텐츠 생성: {text}"
    elif language == 'ru':
        return f"Создать контент на русском: {text}"
    elif language == 'ar':
        return f"إنشاء محتوى باللغة العربية: {text}"
    else:
        return f"Generate content in English: {text}"

def postprocess_text(text, language):
    """
    Postprocess text based on language.
    """
    # Remove the language-specific prefix if needed
    if language == 'zh':
        return text.replace("用中文生成内容：", "")
    elif language == 'es':
        return text.replace("Generar contenido en español: ", "")
    elif language == 'fr':
        return text.replace("Générer du contenu en français: ", "")
    elif language == 'de':
        return text.replace("Generiere Inhalte auf Deutsch: ", "")
    elif language == 'ja':
        return text.replace("日本語でコンテンツを生成する：", "")
    elif language == 'ko':
        return text.replace("한국어로 콘텐츠 생성: ", "")
    elif language == 'ru':
        return text.replace("Создать контент на русском: ", "")
    elif language == 'ar':
        return text.replace("إنشاء محتوى باللغة العربية: ", "")
    else:
        return text.replace("Generate content in English: ", "")
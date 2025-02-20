from flask import Blueprint, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Create a blueprint
narrative_gen_bp = Blueprint('narrative_gen', __name__)

# Load the custom trained model and tokenizer
model_path = "./custom_deepseek_model"
tokenizer_path = "./custom_deepseek_tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Language mapping
LANGUAGE_MAPPING = {
    'en': 'English',
    'zh': 'Chinese',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ru': 'Russian',
    'ar': 'Arabic'
}

@narrative_gen_bp.route('/generate', methods=['POST'])
def generate_content():
    """
    Generate Content API
    ---
    parameters:
      - name: prompt
        in: body
        type: string
        required: true
        description: Prompt for content generation
      - name: language
        in: body
        type: string
        required: true
        description: Language for content generation (en, zh, es, fr, de, ja, ko, ru, ar)
    responses:
      200:
        description: Generated content
        schema:
          type: object
          properties:
            generated_text:
              type: string
    """
    try:
        # Get input data
        prompt = request.json['prompt']
        language = request.json['language']

        # Validate language
        if language not in LANGUAGE_MAPPING:
            return jsonify({'error': f'Unsupported language: {language}. Supported languages are {list(LANGUAGE_MAPPING.keys())}'}), 400

        # Preprocess prompt based on language
        prompt = preprocess_text(prompt, language)

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=100,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )

        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Postprocess text based on language
        generated_text = postprocess_text(generated_text, language)

        return jsonify({'generated_text': generated_text}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

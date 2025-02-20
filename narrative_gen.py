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

@narrative_gen_bp.route('/generate', methods=['POST'])
def generate_content():
    """
    Generate content API
    ---
    parameters:
      - name: prompt
        in: body
        type: string
        required: true
        description: Prompt for generating content
      - name: language
        in: body
        type: string
        required: true
        description: Language for generating content (en, zh, es, etc.)
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

        # Adjust prompt based on language
        if language == 'zh':
            prompt = f"Generate content in Chinese: {prompt}"
        elif language == 'es':
            prompt = f"Generate content in Spanish: {prompt}"
        else:
            prompt = f"Generate content in English: {prompt}"

        # Tokenize
        inputs = tokenizer.encode(prompt, return_tensors="pt")

        # Generate text
        outputs = model.generate(
            inputs,
            max_length=100,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            do_sample=True,
        )

        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({'generated_text': generated_text}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

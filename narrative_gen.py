from flask import Blueprint, request, jsonify, Response
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import json
from typing import Generator, Optional

narrative_gen_bp = Blueprint('narrative_gen', __name__)

# 模型和分词器路径
MODEL_PATH = "./custom_deepseek_model"
TOKENIZER_PATH = "./custom_deepseek_tokenizer"

# 检查GPU可用性并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
model.eval()  # 设置为评估模式以优化推理

# 语言映射
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

# 生成参数配置
GENERATION_CONFIG = {
    "max_new_tokens": 200,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "no_repeat_ngram_size": 3,
    "do_sample": True,
    "num_beams": 1
}

def preprocess_text(text: str, language: str) -> str:
    """根据语言预处理文本"""
    prompts = {
        'zh': "用中文生成内容：",
        'es': "Generar contenido en español: ",
        'fr': "Générer du contenu en français: ",
        'de': "Generiere Inhalte auf Deutsch: ",
        'ja': "日本語でコンテンツを生成する：",
        'ko': "한국어로 콘텐츠 생성: ",
        'ru': "Создать контент на русском: ",
        'ar': "إنشاء محتوى باللغة العربية: ",
        'en': "Generate content in English: "
    }
    return prompts.get(language, prompts['en']) + text

def postprocess_text(text: str, language: str) -> str:
    """根据语言后处理文本"""
    prefixes = {
        'zh': "用中文生成内容：",
        'es': "Generar contenido en español: ",
        'fr': "Générer du contenu en français: ",
        'de': "Generiere Inhalte auf Deutsch: ",
        'ja': "日本語でコンテンツを生成する：",
        'ko': "한국어로 콘텐츠 생성: ",
        'ru': "Создать контент на русском: ",
        'ar': "إنشاء محتوى باللغة العربية: ",
        'en': "Generate content in English: "
    }
    return text.replace(prefixes.get(language, prefixes['en']), "")

def stream_generate(prompt: str, language: str) -> Generator[str, None, None]:
    """流式生成文本"""
    # 设置分词器参数
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 编码输入
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    ).to(device)

    # 创建流式生成器
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.no_grad(), torch.cuda.amp.autocast():  # 使用混合精度加速
        # 生成参数
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "pad_token_id": tokenizer.pad_token_id,
            "streamer": streamer,
            **GENERATION_CONFIG
        }

        # 启动生成过程
        def generate_with_yield():
            output_ids = model.generate(**generation_kwargs)
            for i in range(inputs.input_ids.shape[-1], output_ids.shape[-1]):
                token = tokenizer.decode(output_ids[0][i], skip_special_tokens=True)
                if token:
                    yield token

        return generate_with_yield()

@narrative_gen_bp.route('/generate', methods=['POST'])
def generate_content() -> Response:
    """处理生成内容的API请求"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        prompt = data.get('prompt', '')
        language = data.get('language', 'en')

        if language not in LANGUAGE_MAPPING:
            return jsonify({'error': f'Unsupported language: {language}'}), 400

        # 预处理提示
        processed_prompt = preprocess_text(prompt, language)

        # 生成流式响应
        def generate_stream() -> Generator[str, None, None]:
            for token in stream_generate(processed_prompt, language):
                yield json.dumps({'token': token}, ensure_ascii=False) + '\n'

        return Response(
            generate_stream(),
            mimetype='application/x-ndjson',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        )

    except Exception as e:
        import traceback
        print("Error details:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
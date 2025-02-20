from flask import Flask
from flasgger import Swagger
from signal_filtering import signal_filtering_bp
from blockchain import blockchain_bp
from adaptive_learning import adaptive_learning_bp
#from narrative_gen import narrative_gen_bp #Need to load the trained specific model and wait for subsequent deployment

# Initialize Flask application
app = Flask(__name__)

# Configure Swagger
app.config['SWAGGER'] = {
    'title': 'Unified API Documentation',
    'description': 'API documentation for all features including Signal Filtering, Blockchain, Adaptive Learning, and Narrative Generation.',
    'version': '1.0.0',
    'uiversion': 3
}
swagger = Swagger(app)

# Register blueprints
app.register_blueprint(signal_filtering_bp, url_prefix='/signal_filtering')
app.register_blueprint(blockchain_bp, url_prefix='/blockchain')
app.register_blueprint(adaptive_learning_bp, url_prefix='/adaptive_learning')
#app.register_blueprint(narrative_gen_bp, url_prefix='/narrative_gen') #Need to load the trained specific model and wait for subsequent deployment

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500)
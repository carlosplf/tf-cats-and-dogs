from flask import Flask
from flask_cors import CORS
from api.ml_router import ml_router

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

app.register_blueprint(ml_router)

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import json
import numpy as np
import pandas as pd
import re
import nltk

# ==================================
# NLTK SETUP
# ==================================
nltk_data_path = "/opt/render/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import PorterStemmer

# ==================================
# ONNX MODEL LOAD
# ==================================
print("Current working dir:", os.getcwd())
print("Files in models folder:", os.listdir("models") if os.path.exists("models") else "No models folder")

import onnxruntime as ort

MODEL_PATH = "models/model.onnx"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

onnx_session = ort.InferenceSession(MODEL_PATH)
print("✅ ONNX model loaded successfully")

# ==================================
# TOKENIZER LOAD (plain JSON, no keras)
# ==================================
with open('models/tokenizer.json') as f:
    tokenizer_data = json.load(f)

word_index = tokenizer_data['word_index']
num_words = tokenizer_data['num_words']

def texts_to_sequences(text):
    tokens = text.split()
    return [[word_index[w] for w in tokens if w in word_index and word_index[w] < num_words]]

# ==================================
# FLASK APP SETUP
# ==================================
from datetime import datetime
from blockchain import Blockchain

app = Flask(__name__)
blockchain = Blockchain()

app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "your-secret-key-change-this")
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL", "sqlite:///users.db")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ==================================
# DATABASE MODELS
# ==================================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    analyses = db.relationship('Analysis', backref='user', lazy=True, cascade='all, delete-orphan')

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    review = db.Column(db.Text, nullable=False)
    result = db.Column(db.Integer, nullable=False)  # 0: Genuine, 1: Fake
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    total_fake = db.Column(db.Integer, default=0)
    total_genuine = db.Column(db.Integer, default=0)
try:
    with app.app_context():
        db.create_all()
        print("✅ Database tables created")
        if not User.query.filter_by(username="admin").first():
            db.session.add(User(username="admin", password=generate_password_hash("1234")))
            db.session.commit()
            print("✅ Default user created: admin / 1234")
except Exception as e:
    print(f"⚠️ Startup DB error (non-fatal): {e}")

# ==================================
# HELPERS
# ==================================
def pad_sequences(sequences, maxlen):
    padded = []
    for seq in sequences:
        if len(seq) > maxlen:
            padded.append(seq[-maxlen:])
        else:
            padded.append([0] * (maxlen - len(seq)) + seq)
    return np.array(padded)

MAX_LEN = 150
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = wordpunct_tokenize(text)
    cleaned = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(cleaned)

def predict_review(text):
    clean = preprocess_text(text)
    seq = texts_to_sequences(clean)
    pad = pad_sequences(seq, maxlen=MAX_LEN).astype(np.int32)
    input_name = onnx_session.get_inputs()[0].name
    prob = float(onnx_session.run(None, {input_name: pad})[0][0][0])
    if prob > 0.5:
        return 1, prob
    else:
        return 0, 1 - prob

# ==================================
# USER LOADER
# ==================================
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# ==================================
# ROUTES
# ==================================
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        if password != confirm_password:
            return render_template('login.html', error='Passwords do not match')
        if User.query.filter_by(username=username).first():
            return render_template('login.html', error='Username already exists')
        db.session.add(User(username=username, password=generate_password_hash(password)))
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    analyses = Analysis.query.filter_by(user_id=current_user.id).all()
    total_analyses = len(analyses)
    total_fake = sum(1 for a in analyses if a.result == 1)
    total_genuine = sum(1 for a in analyses if a.result == 0)
    try:
        blockchain_data = blockchain.chain
    except Exception as e:
        print("Blockchain error:", e)
        blockchain_data = []
    return render_template(
        'dashboard.html',
        analyses=analyses,
        total_analyses=total_analyses,
        total_fake=total_fake,
        total_genuine=total_genuine,
        blockchain=blockchain_data
    )

@app.route('/analysis')
@login_required
def analysis():
    return render_template('analysis.html')

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    try:
        review = request.form.get('review', '').strip()
        if not review:
            return jsonify({'error': 'Review cannot be empty'}), 400

        prediction, confidence = predict_review(review)

        if prediction == 0:
            blockchain.add_review({
                'user': current_user.username,
                'review': review,
                'confidence': float(confidence)
            })

        analysis = Analysis(
            user_id=current_user.id,
            review=review,
            result=prediction,
            confidence=confidence
        )
        db.session.add(analysis)
        db.session.commit()

        # Query AFTER commit for accurate counts
        analyses = Analysis.query.filter_by(user_id=current_user.id).all()
        total_fake = sum(1 for a in analyses if a.result == 1)
        total_genuine = sum(1 for a in analyses if a.result == 0)

        return jsonify({
            'result': 'Fake' if prediction == 1 else 'Genuine',
            'confidence': round(confidence * 100, 2),
            'total_fake': total_fake,
            'total_genuine': total_genuine,
            'analysis_id': analysis.id
        })
    except Exception as e:
        print("ERROR IN /api/predict:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
@login_required
def api_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            return jsonify({'error': 'Only CSV and Excel files are supported'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
        results = []

        for idx, row in df.iterrows():
            review_text = str(row.iloc[0])
            prediction, confidence = predict_review(review_text)
            if prediction == 0:
                blockchain.add_review({
                    'user': current_user.username,
                    'review': review_text,
                    'confidence': float(confidence)
                })
            db.session.add(Analysis(
                user_id=current_user.id,
                review=review_text,
                result=prediction,
                confidence=confidence
            ))
            results.append({
                'review': review_text[:100] + '...' if len(review_text) > 100 else review_text,
                'result': 'Fake' if prediction == 1 else 'Genuine',
                'confidence': round(confidence * 100, 2)
            })

        db.session.commit()
        os.remove(filepath)  # cleanup uploaded file

        analyses = Analysis.query.filter_by(user_id=current_user.id).all()
        return jsonify({
            'success': True,
            'results': results,
            'total_fake': sum(1 for a in analyses if a.result == 1),
            'total_genuine': sum(1 for a in analyses if a.result == 0)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results/<int:analysis_id>')
@login_required
def results(analysis_id):
    analysis = db.session.get(Analysis, analysis_id)
    if not analysis or analysis.user_id != current_user.id:
        return redirect(url_for('dashboard'))
    all_analyses = Analysis.query.filter_by(user_id=current_user.id).all()
    return render_template(
        'results.html',
        analysis=analysis,
        total_fake=sum(1 for a in all_analyses if a.result == 1),
        total_genuine=sum(1 for a in all_analyses if a.result == 0),
        total_reviews=len(all_analyses),
        history=all_analyses
    )

@app.route('/blockchain')
@login_required
def view_blockchain():
    try:
        blockchain_data = blockchain.chain or []
        is_valid = blockchain.is_chain_valid() if blockchain_data else True
    except Exception as e:
        print("Blockchain error:", e)
        blockchain_data = []
        is_valid = False
    return render_template(
        'blockchain.html',
        blockchain=blockchain_data,
        length=len(blockchain_data),
        valid=is_valid
    )

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ==================================
# RUN
# ==================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
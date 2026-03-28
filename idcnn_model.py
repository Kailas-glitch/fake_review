# ==================================
# 1. IMPORT LIBRARIES
# ==================================
import pandas as pd
import re
import nltk
import numpy as np
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.callbacks import EarlyStopping

# ==================================
# 2. DOWNLOAD NLTK DATA
# ==================================
nltk.download('stopwords')

# ==================================
# 3. LOAD DATASET
# ==================================
data = pd.read_csv("Fake_Reviews_Dataset1.csv", encoding="latin-1")

data = data[['text', 'label']]
data = data.rename(columns={'text': 'review'})

data = data.dropna(subset=['label'])
data['label'] = data['label'].astype(int)
data['review'] = data['review'].fillna("")

print("\nDataset Distribution:")
print(data['label'].value_counts())

# ==================================
# 4. TEXT PREPROCESSING
# ==================================
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)

    tokens = wordpunct_tokenize(text)

    cleaned = [
        stemmer.stem(w)
        for w in tokens
        if w not in stop_words and len(w) > 2
    ]

    return " ".join(cleaned)

data['clean_review'] = data['review'].apply(preprocess_text)

# ==================================
# 5. TOKENIZATION
# ==================================
MAX_WORDS = 20000
MAX_LEN = 150

tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(data['clean_review'])

sequences = tokenizer.texts_to_sequences(data['clean_review'])
X = pad_sequences(sequences, maxlen=MAX_LEN)

y = data['label'].values

print("Input Shape:", X.shape)

# ==================================
# 6. TRAIN TEST SPLIT
# ==================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# ==================================
# 7. CLASS WEIGHTS
# ==================================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))

print("Class Weights:", class_weights)

# ==================================
# 8. BUILD MODEL (FIXED INPUT)
# ==================================
model = Sequential([

    Input(shape=(MAX_LEN,), dtype='int32'),

    Embedding(
        input_dim=MAX_WORDS,
        output_dim=128
    ),

    Conv1D(256, 5, activation='relu'),
    BatchNormalization(),

    Conv1D(128, 3, activation='relu'),
    BatchNormalization(),

    Conv1D(64, 3, activation='relu'),

    GlobalMaxPooling1D(),

    Dropout(0.5),

    Dense(128, activation='relu'),
    Dropout(0.7),

    Dense(64, activation='relu'),

    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==================================
# 9. TRAIN MODEL
# ==================================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=64,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# ==================================
# 10. EVALUATION
# ==================================
y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==================================
# 11. TEST PREDICTIONS
# ==================================
def predict_review(text):
    clean = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([clean])
    pad = pad_sequences(seq, maxlen=MAX_LEN)

    prob = float(model.predict(pad, verbose=0)[0][0])

    label = "Fake" if prob > 0.5 else "Genuine"
    confidence = prob if prob > 0.5 else 1 - prob
    return label, round(confidence, 4)

print("\n--- TEST RESULTS ---")

test_reviews = [
    "This product is amazing and works perfectly",
    "Worst product ever scam seller",
    "i7t9uh uifhtnki"
]

for r in test_reviews:
    print(r, "→", predict_review(r))

# ==================================
# 12. SAVE TOKENIZER + CONVERT TO ONNX
# ==================================
import os
import subprocess
import numpy as np

os.makedirs("models", exist_ok=True)

# Replace the tokenizer save block with this:
import json
os.makedirs("models", exist_ok=True)

with open("models/tokenizer.json", "w") as f:
    json.dump({
        'word_index': tokenizer.word_index,
        'num_words': tokenizer.num_words
    }, f)

print("✅ Tokenizer saved successfully!")

saved_model_path = "models/saved_model"
model.export(saved_model_path)
# ✅ FIX: Convert using tf2onnx CLI via subprocess (works on all tf2onnx versions)
result = subprocess.run(
    [
        "python", "-m", "tf2onnx.convert",
        "--saved-model", saved_model_path,
        "--output", "models/model.onnx",
        "--opset", "13"
    ],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.returncode != 0:
    print("❌ ONNX conversion failed:")
    print(result.stderr)
else:
    print("\n✅ ONNX model saved successfully!")
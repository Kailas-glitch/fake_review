from tensorflow.keras.models import load_model

model = load_model("models/fake_review_model.keras")
print("Model loaded successfully!")
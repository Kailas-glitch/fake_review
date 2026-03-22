from tensorflow.keras.models import load_model

model = load_model("models/fake_review_model.h5")
model.save("fake_review_model.keras")

print("Conversion done!")
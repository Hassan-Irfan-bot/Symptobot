pip install fastapi uvicorn
from google.colab import drive
drive.mount('/content/drive')


# server.py

from fastapi import FastAPI
from tensorflow.keras.models import load_model
from pydantic import BaseModel

app = FastAPI()

# Load the trained model
model = load_model("/content/drive/MyDrive/DiseasePredictor/diseasepredictor.h5")

class SymptomInput(BaseModel):
    symptoms: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Disease Prediction API!"}

@app.post("/predict_disease")
async def predict_disease(symptom_input: SymptomInput):
    # Perform prediction using the loaded model
    # Assuming model.predict() returns the predicted disease
    prediction = model.predict([symptom_input.symptoms])[0]
    return {"predicted_disease": prediction}
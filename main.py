pip install fastapi uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = FastAPI()

# Load the trained model
model = load_model("/content/drive/MyDrive/Symtobot/data.h5")
encoder = LabelEncoder()

class SymptomInput(BaseModel):
    symptoms: list

@app.post("/predict")
async def predict(symptom_input: SymptomInput):
    try:
        # Preprocess input symptoms
        input_array = preprocess_input(symptom_input.symptoms, headings)
        
        # Predict disease
        prediction = model.predict(input_array.reshape(1, -1))
        predicted_index = np.argmax(prediction)
        predicted_disease = get_predicted_symptom(predicted_index, encoder.classes_)
        
        return {"predicted_disease": predicted_disease}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

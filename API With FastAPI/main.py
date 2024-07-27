import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd


app = FastAPI()

# Load the pre-trained model from the file
with open('model_ckd.pkl', 'rb') as file:
   model = pickle.load(file)

class PredictionInput(BaseModel):
   Age: int
   Blood_Pressure: int
   Specific_Gravity: float
   Albumin: int
   Sugar: int
   Red_Blood_Cells: str
   Pus_Cell: str
   Pus_Cell_clumps: str
   Bacteria: str
   Blood_Glucose_Random: float
   Blood_Urea: float
   Serum_Creatinine: float
   Sodium: float
   Potassium: float
   Hemoglobin: float
   Packed_Cell_Volume: Optional[int]
   White_Blood_Cell_Count: Optional[int]
   Red_Blood_Cell_Count: Optional[float]
   Hypertension: str
   Diabetes_Mellitus: str
   Coronary_Artery_Disease: str
   Appetite: str
   Pedal_Edema: str
   Anemia: str


@app.post("/predict/")
async def predict(input_data: PredictionInput):
   # Convert input data to DataFrame
   input_dict = input_data.dict()
   df = pd.DataFrame([input_dict])
   
   # Rename columns to match the model's expected column names
   df.columns = [
      'Age', 'Blood Pressure', 'Specific Gravity', 'Albumin', 'Sugar', 
      'Red Blood Cells', 'Pus Cell', 'Pus Cell clumps', 'Bacteria', 
      'Blood Glucose Random', 'Blood Urea', 'Serum Creatinine', 'Sodium', 
      'Potassium', 'Hemoglobin', 'Packed Cell Volume', 'White Blood Cell Count', 
      'Red Blood Cell Count', 'Hypertension', 'Diabetes Mellitus', 
      'Coronary Artery Disease', 'Appetite', 'Pedal Edema', 'Anemia'
   ]

   # Preprocess the data if necessary
   # Example: handle categorical variables
   categorical_columns = [
      'Red Blood Cells', 'Pus Cell', 'Pus Cell clumps', 'Bacteria', 
      'Hypertension', 'Diabetes Mellitus', 'Coronary Artery Disease', 
      'Appetite', 'Pedal Edema', 'Anemia'
   ]
   
   df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
   
   # Ensure the order of columns matches the training data
   # This should be adjusted based on how your model was trained
   feature_columns = model.feature_names_in_  # assuming the model has this attribute
   for col in feature_columns:
      if col not in df.columns:
            df[col] = 0
   df = df[feature_columns]
   
   # Make predictions
   predictions = model.predict(df)
   
   return {"predictions": predictions.tolist()}



@app.get("/")
async def root():
   return {"message": "Welcome to the FastAPI app"}

if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="127.0.0.1", port=8000)
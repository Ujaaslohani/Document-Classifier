from fastapi import FastAPI, UploadFile, File, HTTPException
import joblib
import shutil
import os
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
parent_directory = current_file.parent.parent
sys.path.append(str(parent_directory))

from scripts.ocr_extraction import extract_text
from scripts.extract_invoice import extract_invoice_data

app = FastAPI()

model_path = parent_directory / "models" / "classifier.pkl"

if not model_path.exists():
    print(f"Error: Model file not found at {model_path}")
    print("Please ensure that you have trained the model and saved it in the correct location.")
    model = None
else:
    model = joblib.load(str(model_path))

@app.post("/classify_and_extract")
async def classify_and_extract(file: UploadFile = File(...)):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    file_path = os.path.join(temp_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text(file_path)

    category = model.predict([text])[0]

    extracted_data = extract_invoice_data(file_path,category)
    return {"category": category, "details": extracted_data}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
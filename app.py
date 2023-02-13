import uvicorn
import shutil
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from typing import List

# custom imports
from model import get_model, preprocess_image
from worker import working


# init app
app = FastAPI(debug=True)

# init model
MODEL = get_model()

@app.get("/")
async def root(file: UploadFile = File(...)):
    with open(f'{file.filename}', "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename}


@app.post("/image")
async def upload_image(files: List[UploadFile] = File(...)):
    for img in files:
         with open(f'model/imgs/{img.filename}', "wb") as buffer:
             shutil.copyfileobj(img.file, buffer)

    return {"reply": "Good"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with open(f'model/imgs/{file.filename}', "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # path to the temporary image 
    file_path = f'model/imgs/{file.filename}'

    # preprocessing image for the model
    img = preprocess_image(file_path)

    # predicting for the preprocessed image
    classes = MODEL.predict(img)
    
    # after 10 seconds temporary image that was saved, will be deleted
    working(file_path)

    result = {"Prediction": "This Railway track has no fault"}
    if classes<=0.5:
        result["Prediction"] = "This Railway track has fault"
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)




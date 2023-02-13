import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

import pickle

import uvicorn
import shutil
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from typing import List

# init app
app = FastAPI(debug=True)

#MODEL = tf.keras.models.load_model('model/')

@app.post("/")
async def root(file: UploadFile = File(...)):
    with open(f'{file.filename}', "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename}


@app.post("/image")
async def upload_image(files: List[UploadFile] = File(...)):
    for img in files:
         with open(f'{img.filename}', "wb") as buffer:
             shutil.copyfileobj(img.file, buffer)

    return {"reply": "Good"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with open(f'model/imgs/{file.filename}', "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    model = pickle.load(open("railway.h5", "rb"))

    img = cv2.imread(f'model/imgs/{file.filename}')
    plt.imshow(img)
    img = cv2.resize(img,(300,300))
    img = np.reshape(img,[1,300,300,3])
        
    classes = model.predict(img)
        
    print(classes)
    result = {"Prediction": "This Railway track has no fault"}
    if classes<=0.5:
        result["Prediction"] = "This Railway track has fault"
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)




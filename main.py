import uvicorn
import shutil
from fastapi import FastAPI, UploadFile, File

from typing import List

# custom imports
from ml import get_model, preprocess_image
from worker import working


# init app
app = FastAPI(debug=True)

# init model
MODEL = get_model()

@app.get("/")
async def root():
    return {
        "Get prediction from uploaded file": "/image",
        "Get statistics of previous predictions": "/statistics",
        "Get an architecture of a model": "/about",
        "Post an image and a label answer for checking correctness of the model": "/check",
        "Get a history of previous uploaded images and results": "/history"
        }


@app.post("/image")
async def upload_image(files: List[UploadFile] = File(...)):
    for img in files:
         with open(f'model/imgs/{img.filename}', "wb") as buffer:
             shutil.copyfileobj(img.file, buffer)

    return {"reply": "Good"}

@app.post("/uploadfiles")
async def upload_files(files: List[UploadFile]):
    return {"fs": [file.filename for file in files]}

@app.post("/predict")
async def predict(files: List[UploadFile]):
    for file in files:
        with open(f'model/imgs/{file.filename}', "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # path to the temporary image 
    file_paths = [f'model/imgs/{file.filename}' for file in files]

    # preprocessing image for the model
    imgs = [preprocess_image(file_path) for file_path in file_paths]

    # predicting for the preprocessed image
    classes = [MODEL.predict(img) for img in imgs]
    
    # after 10 seconds temporary image that was saved, will be deleted
    for file_path in file_paths:
        working(file_path)

    for i in classes:
        print(i[0][0])
    
    result = {}
    for i in range(len(classes)):
        result[f"{files[i].filename}"] = "This Railway track has a fault"
        if classes[i][0][0] > 0.5: result[f"{files[i].filename}"] = "This Railway track does not have a fault"
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)




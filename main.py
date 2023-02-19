import uvicorn
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse


from typing import List

# custom imports
from ml import get_model, preprocess_image
from worker import working

import numpy as np

# init app
app = FastAPI(debug=True)

# init model
MODEL = get_model()


@app.post("/image")
async def upload_image(files: List[UploadFile] = File(...)):
    for img in files:
         with open(f'model/imgs/{img.filename}', "wb") as buffer:
             shutil.copyfileobj(img.file, buffer)

    return {"reply": "Good"}

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

@app.get("/about")
async def about():

    stringlist = []
    MODEL.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)

    d = {
        "summary": short_model_summary,
        "weights": np.array(MODEL.get_weights()).shape
    }

    return JSONResponse(content=d) 

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)




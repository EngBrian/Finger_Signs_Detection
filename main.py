from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = FastAPI()
model = load_model("finger_signs_model.h5")
class_names = ['0', '1', '2', '3', '4', '5']

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((64, 64))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0 

    prediction = model.predict(img_array)[0]
    predicted_class = class_names[int(np.argmax(prediction))]

    return JSONResponse({"fingers_detected": predicted_class})


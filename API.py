# API.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
from tensorflow.keras.models import load_model

app = FastAPI()

# Cargar el modelo una sola vez al inicio
model = load_model("model_Mnist_LeNet.h5")

def preprocess_image(file_bytes):
    # Cargar imagen desde bytes y convertir a escala de grises
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    img = img.resize((28, 28))  # Redimensionar
    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

@app.post("/predict-digit/")
async def predict_digit(file: UploadFile = File(...)):
    try:
        print("\n--- Nueva solicitud recibida ---")
        print(f"Archivo recibido: {file.filename}")
        
        contents = await file.read()
        image = preprocess_image(contents)
        
        print("Realizando predicción...")
        prediction = model.predict(image)
        digit = int(np.argmax(prediction))
        
        # Mensaje detallado en consola
        print(f"Predicción completa. Probabilidades por clase: {prediction[0]}")
        print(f"Dígito predicho: {digit}")
        
        return JSONResponse(content={"predicted_digit": digit})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

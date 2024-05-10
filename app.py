from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Esto es la API para buscar tu piso ideal!"}

class Descripcion(BaseModel):  # Clase para definir el modelo de entrada
    texto: str

class Output(BaseModel):  # Clase para definir el modelo de salida
    prediction: str

@app.post("/predict", response_model=Output)
async def predict(data: Descripcion):
    # Procesa el texto recibido
    y_pred = data.texto  # Extrae el texto de los datos de entrada
    prediction = {
        'prediction': y_pred  # Envuelve la predicción (en este caso el texto recibido) en un diccionario
    }
    return prediction  # Devuelve el diccionario como un objeto 'Output'

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

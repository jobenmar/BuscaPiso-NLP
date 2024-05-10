from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import requests
import pandas as pd

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

def ToLlama(description: str):
    # Definición de la solicitud a la API LLAMA
    llama_call = {
        "model": "llama3",
        "prompt": "Provide a JSON object with the fields location, rooms, bathrooms, price, surface and contract, that correspond to the location of an apartment, its number of rooms, number of bathrooms, price, surface and contract type, respectively. The contract type can be 'rent' or 'sale'. Extract the information from the following description: " + description,
        "format": "json",
        "stream": False
    }
    
    # URL de la API LLAMA
    url = "http://localhost:11434/api/generate"
    
    # Realizar la solicitud HTTP y obtener la respuesta
    response = requests.post(url, json=llama_call)
    
    # Asumimos que la respuesta es JSON y contiene un campo 'response'
    response_json = response.json()['response']
    
    # Convertir la respuesta en un objeto JSON si no lo es
    if isinstance(response_json, str):
        response_json = json.loads(response_json)
    
    # Guardar el objeto JSON con los datos de la solicitud
    data = json.dumps(response_json, indent=2)
    
    # Convertir el objeto JSON en un DataFrame
    df = pd.DataFrame([response_json])
    
    return df

def FiltrarPisos(df : pd.DataFrame):
    # Cargamos el archivo limpiado de todos los pisos
    df_pisos = pd.read_pickle("mi_dataframe.pkl")

    min_bedrooms = 2
    max_price = 1500
    min_bathrooms = 1
    min_sqft = 120
    contrato = "alquiler"

    # Filtrar el DataFrame por habitaciones, precio, baño, metros cuadrados y tipo contrato
    pisos_filtrado = df_pisos[(df_pisos['Habitaciones'] >= min_bedrooms) &
                            (df_pisos['Precio'] <= max_price) &
                            (df_pisos['Baños'] >= min_bathrooms) &
                            (df_pisos['Metros cuadrados'] >= min_sqft) &
                            (df_pisos['Contrato'] == contrato)]
        
    return pisos_filtrado

@app.post("/predict", response_model=Output)
async def predict(data: Descripcion):
    # Procesa el texto recibido
    description = data.texto  # Extrae el texto de los datos de entrada

    # Realizamos predicción de llama
    df = ToLlama(description)    

    # Filtrar el conjunto de pisos por las caracteristicas obtenidas de llama
    pisos_filtered = FiltrarPisos(df)
    
    prediction = {
        'prediction': "asdf"  # Envuelve la predicción (en este caso el texto recibido) en un diccionario
    }
    return prediction  # Devuelve el diccionario como un objeto 'Output'

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

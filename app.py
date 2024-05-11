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
    prediction: list

def ToLlama(description: str):
    # Definición de la solicitud a la API LLAMA
    llama_call = {
        "model": "llama3",
        "prompt": "Provide a JSON object with the fields location, rooms, bathrooms, price, surface and contract, that correspond to the location of an apartment, its number of rooms, number of bathrooms, price, surface and contract type, respectively. The contract type can be 'alquiler' or 'venta'. Extract the information from the following description: " + description,
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
    df_pisos = pd.read_pickle("pisos_filtrado.pkl")

    # Si falta alguna información ponemos un 0 (menos en contrato que siempre será 'alquiler' o 'venta')
    min_bedrooms = int(df['rooms'].iloc[0] or 0)
    price = int(df['price'].iloc[0] or 0)
    min_bathrooms = int(df['bathrooms'].iloc[0] or 0)
    min_sqft = int(df['surface'].iloc[0] or 0)
    contrato = df['contract'].iloc[0]
    
    # Filtrar el DataFrame basado en las nuevas condiciones
    pisos_filtered = df_pisos[(df_pisos['Habitaciones'] >= min_bedrooms) &
                            ((df_pisos['Precio'] <= price * 1.25) if price > 0 else True) &  # Solo filtrar por precio si no es 0 y multiplicado por 1.25 para buscar pisos ligeramente mas caros que el precio indicado
                            (df_pisos['Baños'] >= min_bathrooms) &
                            (df_pisos['Metros cuadrados'] >= min_sqft * 0.75) & #Multiplicado por 0.75 para buscar pisos ligeramente más pequeños
                            (df_pisos['Contrato'] == contrato)]
    return pisos_filtered

@app.post("/predict", response_model=Output)
async def predict(data: Descripcion):
    # Procesa el texto recibido
    description = data.texto  # Extrae el texto de los datos de entrada

    # Realizamos predicción de llama
    df = ToLlama(description)    

    # Filtrar el conjunto de pisos por las caracteristicas obtenidas de llama
    pisos_filtered = FiltrarPisos(df)
    pisos_filtered.reset_index(inplace=True)
    # print(pisos_filtered)
    prediction = {
        'prediction': pisos_filtered["index"].head(5).tolist()
    }
    return prediction

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

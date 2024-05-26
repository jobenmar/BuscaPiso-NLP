from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity

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
        "prompt": "Create a JSON object with the fields: rooms, bathrooms, max_price, min_price, max_surface, min_surface, and contract. These fields represent the number of rooms in an apartment, the number of bathrooms, the maximum price, the minimum price, the maximum surface area, the minimum surface area, and the contract type, respectively. The contract field can have the values 'alquiler' (rent) or 'venta' (sale). If it is unclear whether the price is a maximum or minimum, assign it to max_price and set min_price to 0. Similarly, if it is unclear whether the surface area is a maximum or minimum, assign it to min_surface and set max_surface to 0. Extract the relevant information from the following description: " + description,
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

    # Manejamos los posibles errores de las caracteristicas obtenidas con Llama
    def safe_convert_to_int(value):
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0
    
    # Si falta alguna información ponemos un 0 (menos en contrato que siempre será 'alquiler' o 'venta')
    min_bedrooms = safe_convert_to_int(df['rooms'].iloc[0])
    max_price = safe_convert_to_int(df['max_price'].iloc[0])
    min_price = safe_convert_to_int(df['min_price'].iloc[0])
    min_bathrooms = safe_convert_to_int(df['bathrooms'].iloc[0])
    max_sqft = safe_convert_to_int(df['max_surface'].iloc[0])
    min_sqft = safe_convert_to_int(df['min_surface'].iloc[0])
    contrato = df['contract'].iloc[0]
    
    # Filtrar el DataFrame basado en las nuevas condiciones
    pisos_filtered = df_pisos[(df_pisos['Habitaciones'] >= min_bedrooms) &
                            ((df_pisos['Precio'] <= max_price) if max_price > 0 else True) &  # Solo filtrar por máximo precio si no es 0 
                            (df_pisos['Precio'] >= min_price) & 
                            (df_pisos['Baños'] >= min_bathrooms) &
                            ((df_pisos['Metros cuadrados'] <= max_sqft) if max_sqft > 0 else True) &  # Solo filtrar por máximo sqft si no es 0 
                            (df_pisos['Metros cuadrados'] >= min_sqft) & 
                            (df_pisos['Contrato'] == contrato)]
    
    # Poner la url de los pisos como columna del dataframe
    pisos_filtered.reset_index(inplace=True)
    print("-------------------------------------------------------------------")

    print(min_bedrooms)
    print(max_price)
    print(min_price)
    print(min_bathrooms)
    print(max_sqft)
    print(min_sqft)
 
    return pisos_filtered

def OrdenarPorSimilitud(df : pd.DataFrame, descripcion : str):
    tv = TfidfVectorizer()
    svd = TruncatedSVD(n_components=50) # Reduciremos las dimensiones a 50
    lsa = make_pipeline(tv, svd, Normalizer(copy=False))
    lsa_topic_vectors = lsa.fit_transform(df["Descripcion"])

    # Transformar la nueva descripción usando la misma pipeline LSA
    vector_descripcion = lsa.transform([descripcion])

    # Calcular la similitud coseno entre la nueva descripción y todas las descripciones existentes
    similarities = cosine_similarity(vector_descripcion, lsa_topic_vectors)

    # Obtener los índices de las descripciones más similares en orden descendente
    most_similar_indices = similarities.argsort()[0][::-1]

    # Reordenar el DataFrame según los índices de similitud más alta
    df_ordered = df.iloc[most_similar_indices]

    return df_ordered


##
# Endpoint para procesar descripciones de pisos:
#   1. Recibe una descripción textual del piso que se quiere buscar.
#   2. Extrae datos estructurados relevantes de la descripción utilizando Llama3.
#   3. Filtra y recupera pisos de pisos_filtrado.pkl que coincidan con los datos estructurados obtenidos.
#   4. Aplica topic modeling (LSA) para evaluar la similitud entre la descripción proporcionada
#      y las descripciones de los pisos filtrados.
#   5. Devuelve los pisos ordenados por su grado de similitud con respecto a la descripción inicial.
##
@app.post("/predict", response_model=Output)
async def predict(data: Descripcion):
    # Procesa el texto recibido
    description = data.texto  # Extrae el texto de los datos de entrada

    # Realizamos predicción de llama
    df = ToLlama(description)    

    # Filtrar el conjunto de pisos por las caracteristicas obtenidas de llama
    pisos_filtered = FiltrarPisos(df)
    
    # Ordenar por similitud en la descripcion
    pisos_ordenados = OrdenarPorSimilitud(pisos_filtered, description)

    # Enviar los 5 pisos más similares
    prediction = {
        'prediction': pisos_ordenados["index"].head(5).tolist()
    }
    return prediction

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

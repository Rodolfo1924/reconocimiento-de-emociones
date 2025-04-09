# YA ESTA EJECUTADO ESTE CODIGO ---- YA NO SE CORRE, SOLO SE VERIFICA EN verificar_conexion.py
import os
import json
from pymongo import MongoClient
from dotenv import load_dotenv
from bson import json_util  # <--- IMPORTANTE

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("DB_NAME")]

folder = "bd"
archivos = os.listdir(folder)
print("Archivos encontrados:", archivos)

for archivo in archivos:
    nombre_coleccion = archivo.replace(f"{os.getenv('DB_NAME')}.", "").replace(".json", "")
    coleccion = db[nombre_coleccion]

    with open(os.path.join(folder, archivo), "r") as f:
        data = json_util.loads(f.read())  # <--- AQUÍ usamos json_util.loads

    # Si es un solo documento, convertirlo en lista
    if isinstance(data, dict):
        data = [data]

    coleccion.insert_many(data)
    print(f"✅ Colección '{nombre_coleccion}' importada correctamente.")



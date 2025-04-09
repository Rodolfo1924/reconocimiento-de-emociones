from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("DB_NAME")]

# Lista todas las colecciones
print("Colecciones en la base de datos:", db.list_collection_names())

# Muestra un documento de 'estimulos'
print("\nEjemplo documento de 'estimulos':")
print(db.estimulos.find_one())


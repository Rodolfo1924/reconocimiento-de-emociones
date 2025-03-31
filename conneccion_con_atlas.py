from pymongo import MongoClient
from bson import DBRef
import datetime

# Conectar a MongoDB (local o Atlas)
client = MongoClient("mongodb://localhost:27017/")  # Si usas Atlas, cambia esta URL

# Crear la base de datos
db = client["proyectos_db"]

# Crear las colecciones
usuarios = db["usuarios"]
proyectos = db["proyectos"]
eventos = db["eventos"]
muestras = db["muestras"]
resultados = db["resultados"]
notificaciones = db["notificaciones"]
comentarios = db["comentarios"]
sesiones = db["sesiones"]

# Crear índices para optimizar las consultas
usuarios.create_index("email", unique=True)
proyectos.create_index("nombre", unique=True)
resultados.create_index("expresion_detectada")

# Insertar un usuario de prueba
usuario_id = usuarios.insert_one({
    "nombre": "Juan Pérez",
    "email": "juan.perez@example.com",
    "telefono": "555-1234",
    "rol": "Investigador",
    "fecha_registro": datetime.datetime.utcnow()
}).inserted_id

# Insertar un proyecto de prueba
proyecto_id = proyectos.insert_one({
    "nombre": "Monitoreo Facial",
    "descripcion": "Proyecto de IA para analizar expresiones faciales en tiempo real.",
    "estatus": "En desarrollo",
    "fecha_creacion": datetime.datetime.utcnow(),
    "responsable": DBRef("usuarios", usuario_id)
}).inserted_id

# Insertar una muestra de prueba
muestra_id = muestras.insert_one({
    "proyecto_id": DBRef("proyectos", proyecto_id),
    "usuario_id": DBRef("usuarios", usuario_id),
    "tipo": "imagen",
    "ruta": "/data/images/img001.jpg",
    "fecha_subida": datetime.datetime.utcnow()
}).inserted_id

# Insertar un resultado de análisis de la IA
resultados.insert_one({
    "muestra_id": DBRef("muestras", muestra_id),
    "expresion_detectada": "Felicidad",
    "confianza": 0.92,
    "fecha_analisis": datetime.datetime.utcnow()
})

# Insertar un evento (ejemplo: usuario sube una imagen)
eventos.insert_one({
    "proyecto_id": DBRef("proyectos", proyecto_id),
    "accion": "Carga de imagen",
    "fecha": datetime.datetime.utcnow(),
    "usuario_id": DBRef("usuarios", usuario_id)
})

# Insertar una notificación
notificaciones.insert_one({
    "usuario_id": DBRef("usuarios", usuario_id),
    "mensaje": "Nuevo resultado disponible",
    "leido": False,
    "fecha_envio": datetime.datetime.utcnow()
})

# Insertar un comentario en un resultado
comentarios.insert_one({
    "usuario_id": DBRef("usuarios", usuario_id),
    "texto": "El resultado es consistente con la hipótesis.",
    "fecha": datetime.datetime.utcnow()
})

# Insertar una sesión de usuario
sesiones.insert_one({
    "usuario_id": DBRef("usuarios", usuario_id),
    "inicio_sesion": datetime.datetime.utcnow(),
    "fin_sesion": None  # Se actualizará cuando cierre sesión
})

print("Base de datos creada en MongoDB Compass con datos de prueba.")

# Usa una imagen base oficial de Python
FROM python:3.12-slim

# Instala las dependencias de sistema que OpenCV necesita
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copia los archivos del proyecto
WORKDIR /app
COPY . /app

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto 8080
EXPOSE 8080

# Comando para correr tu app con gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]

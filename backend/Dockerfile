FROM python:3.12-slim

# Instalar libGL para OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Crear y usar un directorio
WORKDIR /app

# Copiar todo
COPY . .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Comando para iniciar la app
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]

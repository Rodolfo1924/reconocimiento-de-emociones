import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import json
from datetime import datetime
from sys import exit

# Clases de emociones
clases = ['enojado', 'disgusto', 'ansiedad', 'agusto', 'neutral', 'triste', 'sorprendido']

# Obtener ruta base segura (para PyInstaller o desarrollo)
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

ruta_json = os.path.join(base_path, 'backend', 'emociones.json')
modelo_path = os.path.join(base_path, 'backend', 'emotion_model_ligero4.pth')

# Crear emociones.json si no existe, con una lista vacía
if not os.path.exists(ruta_json):
    with open(ruta_json, 'w') as f:
        json.dump([], f)

# Modelo
class NanoEmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(NanoEmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16)
        x = self.fc(x)
        return x

# Cargar modelo
device = torch.device('cpu')
modelo = NanoEmotionCNN(num_classes=len(clases))
modelo.load_state_dict(torch.load(modelo_path, map_location=device))
modelo.eval()

# Transformaciones de imagen
transformar = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Capturar imagen
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error al capturar la imagen")
    exit()

# Procesar imagen
imagen = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
imagen = cv2.resize(imagen, (48, 48))
imagen = transformar(imagen).unsqueeze(0)

# Predecir emoción
with torch.no_grad():
    salida = modelo(imagen)
    prediccion = torch.argmax(salida, dim=1).item()
    emocion = clases[prediccion]
    print(f"Emoción detectada: {emocion}", flush=True)

# Guardar en emociones.json
registro = {
    "timestamp": datetime.now().isoformat(),
    "emocion": emocion
}

# Leer y guardar
try:
    with open(ruta_json, 'r+') as f:
        try:
            datos = json.load(f)
        except json.JSONDecodeError:
            datos = []
        datos.append(registro)
        f.seek(0)
        json.dump(datos, f, indent=2)
except FileNotFoundError:
    with open(ruta_json, 'w') as f:
        json.dump([registro], f, indent=2)
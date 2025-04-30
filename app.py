from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import cv2
import numpy as np
from ultralytics import YOLO
from flask_cors import CORS
from flask import Flask, request, jsonify
from db import usuarios_col
import os

#------------------- front --------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://reconocimiento-de-emociones-f4f7.vercel.app/"}})

# ------------------ CONFIGURACIÓN ------------------
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ MODELO DE EMOCIONES ------------------
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ------------------ MODELO DE EMOCIONES ------------------
emotion_classes = ['Felicidad', 'Enojo', 'Tristeza', 'Sorpresa', 'Neutral']  # <-- 5 clases
emotion_model = EmotionCNN(num_classes=len(emotion_classes))
emotion_model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
emotion_model.to(device).eval()


# ------------------ MODELO YOLO PERSONALIZADO ------------------
yolo_model = YOLO('modelyolov5n.pt')
yolo_model.conf = 0.4  # Umbral de confianza (ajusta si hay falsos positivos)

# ------------------ TRANSFORMACIÓN DE IMAGEN PARA EMOTIONCNN ------------------
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# ------------------ ENDPOINT ------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se recibió archivo'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # YOLO detección
    results = yolo_model(img_cv)
    detecciones = results.xyxy[0].cpu().numpy()

    emociones_detectadas = []

    for det in detecciones:
        x1, y1, x2, y2, _, _ = map(int, det[:6])
        rostro = img_cv[y1:y2, x1:x2]

        if rostro.size == 0:
            continue

        rostro_pil = Image.fromarray(cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)).convert("L")
        tensor = transform(rostro_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            salida = emotion_model(tensor)
            pred = torch.argmax(salida, dim=1).item()
            emociones_detectadas.append(emotion_classes[pred])

    return jsonify({
        "emociones": emociones_detectadas if emociones_detectadas else ["No se detectaron rostros"]
    })

#------------------- bd y login ----------------

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Faltan campos"}), 400

    usuario = usuarios_col.find_one({"email": email})

    if not usuario or usuario.get("password") != password:
        return jsonify({"error": "Credenciales inválidas"}), 401

    return jsonify({"message": "Login exitoso", "usuario": usuario["nombre"]})


@app.route("/register", methods=["POST"])
def register():
    data = request.json
    nombre = data.get("nombre")
    email = data.get("email")
    password = data.get("password")

    if not nombre or not email or not password:
        return jsonify({"error": "Todos los campos son requeridos"}), 400

    if usuarios_col.find_one({"email": email}):
        return jsonify({"error": "Correo ya registrado"}), 409

    usuarios_col.insert_one({
        "nombre": nombre,
        "email": email,
        "password": password  # En producción, usa hash
    })

    return jsonify({"message": "Registro exitoso"}), 201

# ------------------ INICIAR ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

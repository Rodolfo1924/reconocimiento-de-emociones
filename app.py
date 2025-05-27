from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from pymongo import MongoClient
import bcrypt

# Inicializa app
app = Flask(__name__)
CORS(app, supports_credentials=True, origins="*")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Etiquetas de emociones
clases = ['enojado', 'disgusto', 'ansiedad', 'agusto', 'neutral', 'triste', 'sorprendido']

# Modelo ligero
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

# Cargar modelo entrenado
model = NanoEmotionCNN(num_classes=len(clases)).to(device)
model.load_state_dict(torch.load("emotion_model_ligero4.pth", map_location=device))
model.eval()

# Transformación de imagen
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# Conexión a MongoDB Atlas
MONGO_URI = os.environ.get('MONGO_URI')  # pon esto en Render > Environment variables
client = MongoClient(MONGO_URI)
db = client.expresia
users_collection = db.users

# Ruta home
@app.route('/')
def home():
    return jsonify({"message": "API de reconocimiento de emociones y login lista"})

# Ruta de predicción
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se recibió ninguna imagen'}), 400

    try:
        img = Image.open(request.files['file']).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, pred = torch.max(outputs, 1)
            emocion = clases[pred.item()]

        return jsonify({'emocion': emocion})

    except Exception as e:
        return jsonify({'error': f'Error procesando imagen: {str(e)}'}), 500

# Ruta de registro
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if users_collection.find_one({'email': email}):
        return jsonify({'message': 'El usuario ya existe.'}), 400

    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users_collection.insert_one({'email': email, 'password': hashed_pw})

    return jsonify({'message': 'Usuario registrado con éxito.'}), 201

# Ruta de login
@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'message': 'Faltan email o contraseña.'}), 400

    user = users_collection.find_one({'email': email})
    if not user:
        return jsonify({'message': 'Usuario no encontrado.'}), 404

    # Aquí asumimos que user['password'] está guardado como bytes
    if bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return jsonify({'message': 'Inicio de sesión exitoso.'}), 200
    else:
        return jsonify({'message': 'Contraseña incorrecta.'}), 401

# Ejecutar servidor
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

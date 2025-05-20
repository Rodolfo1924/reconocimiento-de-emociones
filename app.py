from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import gdown

# Inicializa app Flask y CORS
app = Flask(__name__)
CORS(app)

# Dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Clases detectadas
clases = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Define arquitectura ligera
class EmotionNet(nn.Module):
    def __init__(self):
        super(EmotionNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 48 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(clases))
        )

    def forward(self, x):
        return self.fc(x)

# Modelo path y descarga si no existe
model_path = 'emotion_model_ligero.pth'
if not os.path.exists(model_path):
    print("Descargando modelo desde Google Drive...")
    url = "https://drive.google.com/uc?id=1tmARiH54eT78OAEP8RoRzjG-KAE25QY3"
    gdown.download(url, model_path, quiet=False)

# Cargar modelo
model = EmotionNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

# Transformación de imagen
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# Ruta raíz
@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "API de reconocimiento de emociones activa."})

# Endpoint /predict
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se recibió imagen'}), 400

    file = request.files['file']
    try:
        img = Image.open(file.stream).convert('RGB')
    except:
        return jsonify({'error': 'Archivo no es una imagen válida'}), 400

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        emocion = clases[pred.item()]

    return jsonify({'emocion': emocion})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

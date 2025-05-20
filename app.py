from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gdown
import os

# Inicializa app Flask y CORS
app = Flask(__name__)
CORS(app)

# Dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Clases detectadas
clases = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Modelo CNN original
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Descargar modelo si no existe
model_path = 'modelo_ligero.pth'
if not os.path.exists(model_path):
    url = 'https://drive.google.com/uc?id=1tmARiH54eT78OAEP8RoRzjG-KAE25QY3'
    gdown.download(url, model_path, quiet=False)

# Cargar modelo
model = EmotionCNN(num_classes=len(clases)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Transformación de imagen
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
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

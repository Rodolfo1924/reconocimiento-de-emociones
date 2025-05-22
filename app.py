from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Inicializa app
app = Flask(__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Etiquetas
clases = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Modelo ultra ligero
class NanoEmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(NanoEmotionCNN, self).__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))  # (4, 48, 48)
        x = self.pool(x)             # (4, 1, 1)
        x = x.view(-1, 4)            # (batch_size, 4)
        x = self.fc(x)
        return x

# Carga directa del modelo ya subido (NO usar gdown en Render Free)
model = NanoEmotionCNN(num_classes=len(clases)).to(device)
model.load_state_dict(torch.load("emotion_model_ligero2.pth", map_location=device))
model.eval()

# Transformación de imagen
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

@app.route('/')
def home():
    return jsonify({"message": "API de reconocimiento de emociones lista"})

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

# Ejecutar localmente
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

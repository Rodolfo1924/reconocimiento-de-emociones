from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__)

# --------- Clase real usada al entrenar el modelo ---------
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

# --------- Cargar modelo entrenado ---------
num_classes = 5  
modelo = EmotionCNN(num_classes)
modelo.load_state_dict(torch.load("emotion_model.pth", map_location=torch.device("cpu")))
modelo.eval()

# --------- Transformación de imagen ---------
transform = transforms.Compose([
    transforms.Grayscale(),            # Porque entrenaste en escala de grises (1 canal)
    transforms.Resize((48, 48)),       # Ajusta si entrenaste con otro tamaño
    transforms.ToTensor()
])

# --------- Endpoint de predicción ---------
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se recibió imagen'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert("L")  # L = grayscale

    input_tensor = transform(image).unsqueeze(0)  # [1, 1, 48, 48]

    with torch.no_grad():
        output = modelo(input_tensor)
        pred = torch.argmax(output, dim=1).item()

    emociones = ['Felicidad', 'Enojo', 'Tristeza', 'Sorpresa']  # Personaliza según tu dataset
    return jsonify({
        'prediction': emociones[pred],
        'class_index': pred
    })

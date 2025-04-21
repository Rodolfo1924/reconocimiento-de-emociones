from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__)

# Cargar el modelo al iniciar el servidor
modelo = torch.load('model.pt', map_location=torch.device('cpu'))
modelo.eval()

# Preprocesamiento de imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = modelo(input_tensor)
        pred = torch.argmax(output, dim=1).item()

    return jsonify({'prediction': int(pred)})
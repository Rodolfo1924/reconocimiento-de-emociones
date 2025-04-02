import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# 🔹 Definir la arquitectura del modelo (debe coincidir con el entrenamiento)
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

# 🔹 Cargar el modelo
num_classes = 7  # Ajusta según las clases de tu dataset
model = EmotionCNN(num_classes)
model.load_state_dict(torch.load("emotion_model.pth", map_location=torch.device('cpu')))
model.eval()

# 🔹 Transformaciones para la imagen de entrada
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 🔹 Etiquetas de clases (ajusta según tu dataset)
class_labels = ["Feliz", "Triste", "Enojado", "Sorprendido", "Neutral", "Asustado", "Disgustado"]

# 🔹 Función para predecir una imagen
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: No se encontró la imagen en {image_path}")
        return
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Añadir dimensión de batch
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_label = class_labels[predicted.item()]
        print(f"Predicción: {predicted_label}")

# 🔹 Prueba con una imagen de ejemplo
image_path = "test_image.jpg"  # Reemplaza con la ruta de tu imagen de prueba
predict_image(image_path)

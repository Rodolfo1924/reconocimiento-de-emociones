import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Definir la arquitectura del modelo (debe coincidir con la que usaste para entrenarlo)
class EmotionCNN(torch.nn.Module):
    def __init__(self, num_classes=7):  # Ajusta num_classes según tus emociones
        super(EmotionCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(128 * 6 * 6, 256)
        self.fc2 = torch.nn.Linear(256, num_classes)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Clases de emociones (ajústalas según tu dataset)
emotion_classes = ['Feliz', 'Triste', 'Enojado', 'Sorprendido', 'Neutral', 'Miedo', 'Asco']

# Cargar el modelo entrenado
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(num_classes=len(emotion_classes)).to(device)
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.eval()

# Transformaciones para la imagen
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convertir a escala de grises
    transforms.Resize((48, 48)),  # Redimensionar a 48x48
    transforms.ToTensor(),  # Convertir a tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalizar
])

# Función para hacer una predicción
def predict_emotion(image_path):
    if not os.path.exists(image_path):
        print("Error: La imagen no existe.")
        return

    image = Image.open(image_path)  # Cargar imagen
    image = transform(image).unsqueeze(0).to(device)  # Aplicar transformaciones y agregar dimensión batch
    
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    print(f"Emoción detectada: {emotion_classes[prediction]}")

# Ruta de la imagen de prueba (ajústala según tu archivo)
image_path = "test_image.jpg"  # Cambia esto por la ruta de tu imagen
predict_emotion(image_path)

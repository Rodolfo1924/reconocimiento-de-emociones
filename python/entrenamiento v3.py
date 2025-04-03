import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os

# 🔹 Verificar si CUDA está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# 🔹 Configurar rutas del dataset
dataset_path = "/workspaces/reconocimiento-de-emociones/dataset/images"  
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "validation")

# 🔹 Verificar si las carpetas existen
if not os.path.exists(train_path) or not os.path.exists(val_path):
    raise FileNotFoundError("Error: Las carpetas 'train' o 'validation' no existen. Verifica la estructura del dataset.")

# 🔹 Transformaciones para las imágenes (preprocesamiento)
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convertir a escala de grises
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),  # Aumentación de datos
    transforms.RandomRotation(10),      # Aumentación de datos
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalización
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 🔹 Cargar datasets
train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)

# 🔹 Verificar si las carpetas contienen imágenes
if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise ValueError("Error: No se encontraron imágenes en 'train' o 'validation'. Verifica la estructura.")

# 🔹 Crear DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# 🔹 Definir modelo CNN
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

# 🔹 Inicializar modelo
num_classes = len(train_dataset.classes)  # Detectar número de clases automáticamente
model = EmotionCNN(num_classes).to(device)

# 🔹 Definir función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 🔹 Función para entrenar el modelo
def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # 🔹 Evaluar el modelo en el conjunto de validación
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {100 * correct/total:.2f}%, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {100 * val_correct/val_total:.2f}%")

# 🔹 Entrenar el modelo
train_model(model, train_loader, val_loader, epochs=300w3)

# 🔹 Guardar el modelo entrenado
torch.save(model.state_dict(), "emotion_model.pth")
print("Modelo guardado correctamente.")
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os

# Verificar CUDA
print(f"¿CUDA disponible? {torch.cuda.is_available()}")
print(f"Nombre de la GPU: {torch.cuda.get_device_name(0)}")

# Configuración de dispositivo (GPU si está disponible, si no, CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Directorios de datos
dataset_path = "/workspaces/reconocimiento-de-emociones/dataset/images"  # Ajusta si es necesario
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "validation")

# Transformaciones para normalizar y aumentar los datos
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convertir a escala de grises
    transforms.Resize((48, 48)),  # Redimensionar a 48x48
    transforms.RandomHorizontalFlip(),  # Aumentación de datos: volteo horizontal aleatorio
    transforms.RandomRotation(10),  # Aumentación de datos: rotación aleatoria de ±10 grados
    transforms.ToTensor(),  # Convertir a tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalizar con media 0.5 y desviación 0.5
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Carga del dataset
train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)

# Verificar clases detectadas
num_classes = len(train_dataset.classes)
print(f"Clases detectadas: {train_dataset.classes}")
print(f"Número de imágenes en train: {len(train_dataset)}")
print(f"Número de imágenes en validation: {len(val_dataset)}")

# Crear DataLoader para train y validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Definición del modelo CNN
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Calcular el tamaño de entrada de la capa fully connected
        self._to_linear = self._get_conv_output((1, 48, 48))

        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def _get_conv_output(self, shape):
        """Calcula el tamaño de salida después de las capas convolucionales."""
        x = torch.rand(1, *shape)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, self._to_linear)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Inicializar modelo, función de pérdida y optimizador
model = EmotionCNN(num_classes).to(device)  # Mover el modelo a la GPU si está disponible
criterion = nn.CrossEntropyLoss()  # Función de pérdida
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizador Adam

# Función de entrenamiento
def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()  # Modo de entrenamiento
        running_loss = 0.0
        correct = 0
        total = 0

        # Bucle de entrenamiento
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Mover datos a la GPU
            optimizer.zero_grad()  # Reiniciar los gradientes
            outputs = model(images)  # Pasar las imágenes por el modelo
            loss = criterion(outputs, labels)  # Calcular la pérdida
            loss.backward()  # Retropropagación
            optimizer.step()  # Actualizar los pesos

            # Calcular métricas
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Evaluación en validación
        model.eval()  # Modo de evaluación
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # Desactivar el cálculo de gradientes para validación
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)  # Mover datos a la GPU
                outputs = model(images)  # Pasar las imágenes por el modelo
                loss = criterion(outputs, labels)  # Calcular la pérdida
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # Imprimir métricas
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {100 * correct/total:.2f}%, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {100 * val_correct/val_total:.2f}%")

# Entrenar el modelo
train_model(model, train_loader, val_loader, epochs=10)

# Guardar el modelo entrenado
torch.save(model.state_dict(), "emotion_model.pth")
print("Modelo guardado correctamente.")
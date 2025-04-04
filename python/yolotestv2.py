import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image

# =========================
# DEFINICIÓN DEL MODELO CNN
# =========================

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=5):  # IMPORTANTE: aquí van las 5 clases
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

# =========================
# CARGA DE MODELOS
# =========================

# Cargar YOLO para detección de rostro
face_model = YOLO("yolov11n-face.pt")

# Cargar modelo de emociones entrenado
emotion_model = EmotionCNN(num_classes=5)  # IMPORTANTE: debe coincidir con el modelo entrenado
emotion_model.load_state_dict(torch.load("emotion_model.pth", map_location=torch.device('cpu')))
emotion_model.eval()

# =========================
# TRANSFORMACIONES DE IMAGEN
# =========================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Etiquetas (AJUSTA si tu dataset tiene otras)
class_labels = ["Feliz", "Triste", "Enojado", "Sorprendido", "Neutral"]

# =========================
# FUNCIÓN DE DETECCIÓN DE EMOCIÓN
# =========================

def detectar_emocion(face_img):
    try:
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0)  # Añadir dimensión de batch
        with torch.no_grad():
            output = emotion_model(face_tensor)
            _, predicted = torch.max(output, 1)
        return class_labels[predicted.item()]
    except Exception as e:
        print(f"Error al procesar rostro: {e}")
        return "Desconocido"

# =========================
# CAPTURA EN TIEMPO REAL
# =========================

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detección con YOLO
        results = face_model(frame)

        rostro_detectado = False  # Bandera para controlar detección única

        for r in results:
            for box in r.boxes:
                if rostro_detectado:
                    break  # Salir del bucle interior si ya se procesó un rostro

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                emotion = detectar_emocion(face)

                # Dibujar caja y emoción
                cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 255, 50), 2)
                cv2.putText(frame, emotion, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 255, 50), 2)

                rostro_detectado = True  # Ya se procesó un rostro

            if rostro_detectado:
                break  # Salir también del bucle exterior

        cv2.imshow("Detector de Emociones", frame)

        # Salir si se presiona 'q' o si se cierra la ventana con la X
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty("Detector de Emociones", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================
# EJECUCIÓN
# =========================

if __name__ == "__main__":
    main()

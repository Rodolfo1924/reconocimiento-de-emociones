from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

# Cargar modelo (igual que en local)
class NanoEmotionCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(NanoEmotionCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool2 = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(16, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16)
        x = self.fc(x)
        return x

clases = ['enojado', 'disgusto', 'ansiedad', 'agusto', 'neutral', 'triste', 'sorprendido']
modelo = NanoEmotionCNN(num_classes=len(clases))
modelo.load_state_dict(torch.load('emotion_model_ligero4.pth', map_location='cpu'))
modelo.eval()

transformar = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

@app.post("/detectar")
async def detectar_emocion(file: UploadFile = File(...)):
    contenido = await file.read()
    imagen = Image.open(BytesIO(contenido)).convert('RGB')
    imagen = transformar(imagen).unsqueeze(0)

    with torch.no_grad():
        salida = modelo(imagen)
        prediccion = torch.argmax(salida, dim=1).item()
        emocion = clases[prediccion]

    return JSONResponse(content={"emocion": emocion})

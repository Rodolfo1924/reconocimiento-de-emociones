const express = require('express');
const cors = require('cors');
const connectDB = require('./database');
const authRoutes = require('./routes/auth');
const multer = require('multer'); // Para manejar archivos

const app = express();

// Conectar a MongoDB
connectDB();

// Middleware
app.use(cors());
app.use(express.json());

// Configuración de multer para manejar la subida de archivos
const storage = multer.memoryStorage(); // Almacena el archivo en memoria
const upload = multer({ storage: storage });

app.use('/api/auth', authRoutes);

// Endpoint para predicción
app.post('/predict', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: 'No se proporcionó ningún archivo' });
    }

    // Aquí iría la lógica real de predicción
    // Ejemplo: Procesar la imagen con un modelo de machine learning
    // Por ahora, simulamos una predicción
    const prediction = await simulatePrediction(req.file.buffer); // Ejemplo simulado
    res.status(200).json({ prediction });
  } catch (error) {
    console.error('Error en predicción:', error);
    res.status(500).json({ message: 'Error al predecir emoción', error: error.message });
  }
});

// Función simulada de predicción (reemplázala con tu lógica real)
const simulatePrediction = (imageBuffer) => {
  // Simulación: Devuelve una emoción aleatoria
  const emotions = ['Felicidad', 'Tristeza', 'Enojo', 'Sorpresa'];
  return new Promise((resolve) => {
    setTimeout(() => {
      const randomEmotion = emotions[Math.floor(Math.random() * emotions.length)];
      resolve(randomEmotion);
    }, 1000); // Simula un procesamiento de 1 segundo
  });
};

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Servidor corriendo en el puerto ${PORT}`);
});

module.exports = app; // Exporta app para pruebas si es necesario
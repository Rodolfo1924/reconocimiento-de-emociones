const fs = require('fs');
const axios = require('axios');
const FormData = require('form-data');
const path = require('path');
const { guardarEmocion } = require('./storage');

const BACKEND_URL = 'https://reconocimiento-de-emociones.onrender.com/predict';

async function enviarImagenAlBackend(rutaImagen) {
  try {
    const form = new FormData();
    form.append('image', fs.createReadStream(rutaImagen));

    const response = await axios.post(BACKEND_URL, form, {
      headers: form.getHeaders(),
      timeout: 10000 // opcional
    });

    const emocion = response.data?.emocion || response.data?.emotion || 'desconocida';
    console.log('Emoción detectada:', emocion);

    guardarEmocion({
      emocion,
      ruta: rutaImagen,
      fecha: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error al enviar imagen al backend:', error.message);
  }
}

module.exports = { enviarImagenAlBackend };
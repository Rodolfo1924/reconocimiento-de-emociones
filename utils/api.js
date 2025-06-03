const axios = require('axios');

// Aquí pones la URL pública de tu backend en Render
const API_URL = 'https://reconocimiento-de-emociones.onrender.com';

async function enviarImagenAlBackend(imagenBase64) {
  try {
    const respuesta = await axios.post(`${API_URL}/emociones`, {
      imagen: imagenBase64,
    });
    return respuesta.data;
  } catch (error) {
    console.error('Error consultando API:', error.message);
    return null;
  }
}

module.exports = { enviarImagenAlBackend };

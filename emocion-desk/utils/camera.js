const NodeWebcam = require('node-webcam');
const path = require('path');
const fs = require('fs');
const { enviarImagenAlBackend } = require('./api');

const rutaCapturas = path.join(__dirname, '../capturas');

if (!fs.existsSync(rutaCapturas)) {
  fs.mkdirSync(rutaCapturas);
}

const opciones = {
  width: 640,
  height: 480,
  quality: 80,
  delay: 0,
  saveShots: true,
  output: 'jpeg',
  device: false,
  callbackReturn: 'location',
  verbose: false
};

const Webcam = NodeWebcam.create(opciones);

function capturarImagen() {
  const timestamp = new Date().toISOString().replace(/:/g, '-');
  const nombreArchivo = `captura-${timestamp}.jpg`;
  const rutaImagen = path.join(rutaCapturas, nombreArchivo);

  NodeWebcam.capture(rutaImagen, opciones, (err, data) => {
    if (err) {
      console.error('Error al capturar imagen:', err);
    } else {
      console.log('Imagen capturada:', rutaImagen);
      enviarImagenAlBackend(rutaImagen); // aquí se envía al backend Flask
    }
  });
}

module.exports = { capturarImagen };
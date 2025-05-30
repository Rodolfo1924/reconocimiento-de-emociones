const fs = require('fs');
const path = require('path');

const rutaDirectorio = path.join(__dirname, '../data');
const rutaArchivo = path.join(rutaDirectorio, 'registro-emociones.json');

// Asegura que exista el directorio /data
if (!fs.existsSync(rutaDirectorio)) {
  fs.mkdirSync(rutaDirectorio);
}

// Inicializa el archivo si no existe
if (!fs.existsSync(rutaArchivo)) {
  fs.writeFileSync(rutaArchivo, JSON.stringify([]));
}

function guardarEmocion({ emocion, fecha, ruta }) {
  try {
    const data = JSON.parse(fs.readFileSync(rutaArchivo, 'utf-8'));
    data.push({ emocion, fecha, ruta });
    fs.writeFileSync(rutaArchivo, JSON.stringify(data, null, 2));
    console.log('Emoción guardada:', emocion, 'en', fecha);
  } catch (err) {
    console.error('Error al guardar emoción:', err.message);
  }
}

function obtenerEmocionesDelDia(diaISO = new Date().toISOString().split('T')[0]) {
  try {
    const data = JSON.parse(fs.readFileSync(rutaArchivo, 'utf-8'));
    return data.filter(e => e.fecha.startsWith(diaISO));
  } catch (err) {
    console.error('Error al leer emociones:', err.message);
    return [];
  }
}

module.exports = {
  guardarEmocion,
  obtenerEmocionesDelDia
};
const fs = require('fs');
const path = require('path');
const axios = require('axios');
const { obtenerEmocionesDelDia } = require('./storage');

const rutaReportes = path.join(__dirname, '../reports');
if (!fs.existsSync(rutaReportes)) {
  fs.mkdirSync(rutaReportes);
}

const emocionesPosibles = [
  'feliz', 'triste', 'enojado', 'sorprendido', 'neutral', 'desconocida'
];

function contarEmociones(emociones) {
  const conteo = {};
  emocionesPosibles.forEach(e => (conteo[e] = 0));
  emociones.forEach(({ emocion }) => {
    if (emocionesPosibles.includes(emocion)) conteo[emocion]++;
    else conteo['desconocida']++;
  });
  return conteo;
}

async function generarReporte(diaISO = new Date().toISOString().split('T')[0]) {
  try {
    const emociones = obtenerEmocionesDelDia(diaISO);
    if (emociones.length === 0) {
      console.log('No hay emociones para el día', diaISO);
      return null;
    }

    const conteo = contarEmociones(emociones);
    const etiquetas = Object.keys(conteo);
    const datos = Object.values(conteo);

    // Construir URL para quickchart.io
    const chartConfig = {
      type: 'pie',
      data: {
        labels: etiquetas,
        datasets: [{
          data: datos,
          backgroundColor: [
            '#4caf50', // verde feliz
            '#2196f3', // azul triste
            '#f44336', // rojo enojado
            '#ffeb3b', // amarillo sorprendido
            '#9e9e9e', // gris neutral
            '#607d8b'  // azul oscuro desconocida
          ]
        }]
      },
      options: {
        title: {
          display: true,
          text: `Reporte de emociones del día ${diaISO}`
        }
      }
    };

    const urlChart = `https://quickchart.io/chart?c=${encodeURIComponent(JSON.stringify(chartConfig))}`;

    // Descargar imagen PNG
    const response = await axios.get(urlChart, { responseType: 'arraybuffer' });
    const rutaArchivo = path.join(rutaReportes, `reporte-${diaISO}.png`);
    fs.writeFileSync(rutaArchivo, response.data);

    console.log('Reporte generado:', rutaArchivo);
    return rutaArchivo;

  } catch (error) {
    console.error('Error generando reporte:', error.message);
    return null;
  }
}

module.exports = { generarReporte };

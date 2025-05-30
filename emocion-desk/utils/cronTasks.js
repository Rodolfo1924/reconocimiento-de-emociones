const cron = require('node-cron');
const { generarReporte } = require('./report');

// Programa la tarea para que corra todos los días a las 23:59
function iniciarTareasProgramadas() {
  cron.schedule('59 23 * * *', async () => {
    console.log('Ejecutando tarea programada: generar reporte diario');
    await generarReporte();
  });
}

module.exports = { iniciarTareasProgramadas };

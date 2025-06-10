const { execFile } = require('child_process');
const path = require('path');

const exePath = path.join(__dirname, '..', 'backend', 'detectar_emocion.py');

function ejecutarDeteccion() {
  console.log('Intentando ejecutar:', exePath);
  execFile(exePath, (error, stdout, stderr) => {
    if (error) {
      console.error('Error ejecutando detectar_emocion.exe:', error);
      return;
    }
    if (stdout) console.log('stdout:', stdout);
    if (stderr) console.error('stderr:', stderr);
  });
}

// Ejecutar cada 10 segundos para pruebas
setInterval(ejecutarDeteccion, 10000);

const path = require('path');
const { generarReporte } = require(path.join(__dirname, '../utils/report'));

async function main() {
  await generarReporte();
  process.exit(0);
}

main();

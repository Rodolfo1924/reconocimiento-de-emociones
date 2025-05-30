const { app, BrowserWindow, Tray, Menu, nativeImage } = require('electron');
const path = require('path');
const cron = require('node-cron');
const { capturarImagen } = require('./utils/camera');

let mainWindow;
let tray;

function createWindow() {
mainWindow = new BrowserWindow({
width: 300,
height: 500,
show: false,
skipTaskbar: true,
icon: path.join(__dirname, 'assets/icono.ico'),
webPreferences: {
preload: path.join(__dirname, 'preload.js'),
},
});

mainWindow.loadFile('renderer/index.html');
}

app.whenReady().then(() => {
createWindow();

// Tray icon
const icono = nativeImage.createFromPath(path.join(__dirname, 'assets/icono.ico'));
tray = new Tray(icono);
const contextMenu = Menu.buildFromTemplate([
{ label: 'Mostrar dashboard', click: () => mainWindow.show() },
{ label: 'Salir', click: () => app.quit() },
]);
tray.setToolTip('Reconocimiento de emociones');
tray.setContextMenu(contextMenu);

// Tarea programada: cada 10 minutos
cron.schedule('*/10 * * * *', () => {
capturarImagen();
});
});

app.on('window-all-closed', (e) => {
e.preventDefault(); // Evita que se cierre la app al cerrar la ventana
});
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('emocionesAPI', {
  leerEmociones: () => ipcRenderer.invoke('leer-emociones')
});
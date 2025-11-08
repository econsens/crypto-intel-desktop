const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('api', {
  restartApp: () => ipcRenderer.send('restart_app'),
});

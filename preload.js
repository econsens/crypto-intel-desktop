// Secure preload (we're not exposing Node APIs to the page right now)
const { contextBridge } = require('electron');
contextBridge.exposeInMainWorld('cryptoIntel', {
  version: '1.0.0'
});

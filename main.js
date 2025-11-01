const { app, BrowserWindow, ipcMain, Menu } = require('electron');
const path = require('path');
const { autoUpdater } = require('electron-updater');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    icon: path.join(__dirname, 'assets', 'icon.ico'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  const appVersion = app.getVersion();
  mainWindow.setTitle(`Crypto Intel v${appVersion}`);

  // If /alerts showed "Not Found", point to root "/"
  mainWindow.loadURL('http://127.0.0.1:8000/');

  mainWindow.on('closed', () => (mainWindow = null));
}

const template = [
  { label: 'File', submenu: [{ role: 'quit' }] },
  {
    label: 'View',
    submenu: [{ role: 'reload' }, { role: 'forcereload' }, { type: 'separator' }, { role: 'toggledevtools' }],
  },
  {
    label: 'Help',
    submenu: [
      {
        label: 'About',
        click: () => {
          const { dialog } = require('electron');
          dialog.showMessageBox({
            type: 'info',
            title: 'About',
            message: `Crypto Intel Desktop\nVersion ${app.getVersion()}`,
          });
        },
      },
    ],
  },
];

Menu.setApplicationMenu(Menu.buildFromTemplate(template));

app.on('ready', () => {
  createWindow();
  autoUpdater.checkForUpdatesAndNotify();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (mainWindow === null) createWindow();
});

autoUpdater.on('update-available', () => {
  if (mainWindow) mainWindow.webContents.send('update_available');
});

autoUpdater.on('update-downloaded', () => {
  if (mainWindow) mainWindow.webContents.send('update_downloaded');
});

ipcMain.on('restart_app', () => {
  autoUpdater.quitAndInstall();
});

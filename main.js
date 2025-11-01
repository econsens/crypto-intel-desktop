// ==========================
// Crypto Intel - Electron Main (CJS-safe)
// ==========================
//
// What this file does:
// 1) Ensures the Docker backend container is running (or creates it).
// 2) Opens the app window pointed at http://localhost:8000/.
// 3) Adds a tray menu (Open, Check for Updates, Quit).
// 4) Enables auto-update via GitHub Releases using electron-updater.
//
// Requirements:
// - docker.js in the same folder (with ensureContainerRunning & stopContainer)
// - preload.js in the same folder
// - assets/icon.ico and assets/icon.png exist
// - package.json "publish" configured for your GitHub repo
//
// -------------------------------------------------------------

const { app, BrowserWindow, dialog, Menu, Tray, nativeImage, Notification } = require('electron');
const path = require('path');
const { autoUpdater } = require('electron-updater');
const { ensureContainerRunning, stopContainer } = require('./docker');

// Use Electron's own flag to detect dev vs packaged build
const isDev = !app.isPackaged;

// ---------- Config you may change ----------
const APP_PORT = 8000;                                // Backend FastAPI port
const APP_URL  = `http://localhost:${APP_PORT}/`;     // Frontend URL to load
const CONTAINER_NAME = 'crypto-mini';                 // Docker container name
const DATA_DIR = 'C:\\\\crypto-intel-mini\\\\data';   // Host path for /data in container
const IMAGE_NAME = 'crypto-mini';                     // Docker image tag built from your Dockerfile
// -------------------------------------------

let mainWindow = null;
let tray = null;

// Single instance (avoid multiple running windows)
if (!app.requestSingleInstanceLock()) {
  app.quit();
}
app.on('second-instance', () => {
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.show();
    mainWindow.focus();
  }
});

// Graceful quit: stop/remove container then quit app
async function gracefulQuit() {
  try {
    await stopContainer(CONTAINER_NAME);
  } catch (_) {
    // ignore errors on shutdown
  }
  app.exit(0);
}

// Create main window
function createMainWindow() {
  const iconPng = nativeImage.createFromPath(path.join(__dirname, 'assets', 'icon.png'));
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 1100,
    minHeight: 720,
    backgroundColor: '#0b1220',
    icon: iconPng,
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  });

  mainWindow.loadURL(APP_URL);

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Build tray menu
function setupTray() {
  const trayIcon = path.join(__dirname, 'assets', 'icon.ico');
  tray = new Tray(trayIcon);
  tray.setToolTip('Crypto Intel');

  const menu = Menu.buildFromTemplate([
    {
      label: 'Open Crypto Intel',
      click: () => {
        if (!mainWindow) createMainWindow();
        mainWindow.show();
        mainWindow.focus();
      }
    },
    { type: 'separator' },
    {
      label: 'Check for Updates',
      click: () => {
        try {
          autoUpdater.checkForUpdatesAndNotify();
        } catch (err) {
          dialog.showMessageBox({
            type: 'info',
            title: 'Crypto Intel',
            message: 'Update check failed',
            detail: String(err)
          });
        }
      }
    },
    { type: 'separator' },
    {
      label: 'Quit',
      click: async () => {
        await gracefulQuit();
      }
    }
  ]);

  tray.setContextMenu(menu);
  tray.on('click', () => {
    if (!mainWindow) createMainWindow();
    mainWindow.show();
    mainWindow.focus();
  });
}

// Configure auto-updater
function setupAutoUpdater() {
  // Only enable auto-update when NOT in dev
  if (isDev) return;

  autoUpdater.autoDownload = true;            // download in background
  autoUpdater.autoInstallOnAppQuit = true;    // install on next quit if user ignores prompt

  try {
    const log = require('electron-log');
    autoUpdater.logger = log;
    autoUpdater.logger.transports.file.level = 'info';
  } catch (_) {
    // electron-log is optional
  }

  autoUpdater.on('update-available', (info) => {
    new Notification({ title: 'Crypto Intel', body: `Update ${info.version} is available. Downloading…` }).show();
  });

  autoUpdater.on('download-progress', (progress) => {
    if (mainWindow && !mainWindow.isDestroyed() && progress && progress.percent) {
      mainWindow.setProgressBar(progress.percent / 100);
    }
  });

  autoUpdater.on('update-downloaded', async (info) => {
    if (mainWindow && !mainWindow.isDestroyed()) mainWindow.setProgressBar(-1);
    const res = await dialog.showMessageBox({
      type: 'info',
      title: 'Crypto Intel — Update ready',
      message: `Version ${info.version} has been downloaded.`,
      detail: 'Restart now to install the update?',
      buttons: ['Restart & Install', 'Later'],
      defaultId: 0,
      cancelId: 1
    });
    if (res.response === 0) {
      try { await stopContainer(CONTAINER_NAME); } catch (_) {}
      autoUpdater.quitAndInstall();
    }
  });

  autoUpdater.on('error', (err) => {
    console.log('AutoUpdate error:', err?.message || err);
  });

  try {
    autoUpdater.checkForUpdatesAndNotify();
  } catch (err) {
    console.log('Initial update check failed:', err?.message || err);
  }
}

// App bootstrap
async function start() {
  // 1) Ensure Docker backend is running
  const result = await ensureContainerRunning({
    container: CONTAINER_NAME,
    dataDir: DATA_DIR,
    image: IMAGE_NAME,
    port: APP_PORT
  });

  if (!result.ok) {
    await dialog.showMessageBox({
      type: 'error',
      title: 'Crypto Intel',
      message: 'Failed to start local backend (Docker).',
      detail: (result && result.message) ? result.message : 'Is Docker Desktop running?',
      buttons: ['OK']
    });
    app.quit();
    return;
  }

  // 2) Create main window
  createMainWindow();

  // 3) Create tray menu
  setupTray();

  // 4) Configure auto-updates
  setupAutoUpdater();
}

app.whenReady().then(start);

// Keep app running when all windows closed (tray controls it)
app.on('window-all-closed', () => {
  // On Windows we keep the tray running; do nothing here.
});

// On explicit quit (e.g., system shutdown), stop docker and exit
app.on('before-quit', async () => {
  try { await stopContainer(CONTAINER_NAME); } catch (_) {}
});

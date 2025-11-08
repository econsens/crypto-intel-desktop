// main.js — Crypto Intel (classic layout)
// - Loads your backend UI directly (http://127.0.0.1:8000/)
// - Shows version in the window title
// - Auto-updates from GitHub Releases (electron-updater)
// - Minimal menu with About + Check for Updates

const path = require("path");
const {
  app,
  BrowserWindow,
  Menu,
  dialog,
  ipcMain,
  shell,
} = require("electron");
const { autoUpdater } = require("electron-updater");

// ----- App identity (IMPORTANT: keep in sync with package.json appId) -----
app.setName("Crypto Intel");
app.setAppUserModelId("com.crypto.intel.desktop"); // must match your original appId identity

// ----- Single instance lock (optional but recommended) -----
if (!app.requestSingleInstanceLock()) {
  app.quit();
  process.exit(0);
}
app.on("second-instance", () => {
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.focus();
  }
});

let mainWindow;

// Helpers
function windowTitle() {
  return `Crypto Intel v${app.getVersion()}`;
}
function iconPath() {
  // Windows icon; add .icns/.png if you target macOS/Linux later.
  return path.join(__dirname, "assets", "icon.ico");
}

// ----- Create the main window -----
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 1100,
    minHeight: 700,
    backgroundColor: "#0e1621",
    title: windowTitle(),
    icon: iconPath(),
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  // Keep our title even if the page tries to change it
  mainWindow.on("page-title-updated", (e) => {
    e.preventDefault();
    if (!mainWindow.isDestroyed()) mainWindow.setTitle(windowTitle());
  });

  // Load your backend UI (root path is safest)
  const UI_URL = "http://127.0.0.1:8000/";
  mainWindow.loadURL(UI_URL).catch((err) => {
    // Fallback inline error page if backend is not up yet
    const html = `
      <body style="margin:0;height:100vh;display:flex;align-items:center;justify-content:center;background:#0e1621;color:#cbd5e1;font-family:system-ui,Segoe UI,Roboto,Arial">
        <div style="max-width:720px;text-align:center">
          <h1 style="margin:0 0 12px">Crypto Intel Desktop</h1>
          <p>Could not reach <code>${UI_URL}</code>.</p>
          <p>Make sure your backend container is running, then click Retry.</p>
          <button onclick="location.reload()" style="padding:10px 14px;border-radius:8px;border:0;background:#2563eb;color:#fff;cursor:pointer">Retry</button>
        </div>
      </body>
    `;
    mainWindow.loadURL("data:text/html;charset=utf-8," + encodeURIComponent(html));
    console.error("Failed to load UI:", err);
  });

  // Open external links in the default browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

// ----- Menu (minimal) -----
function buildMenu() {
  const template = [
    {
      label: "File",
      submenu: [{ role: "quit", label: "Exit" }],
    },
    {
      label: "View",
      submenu: [
        { role: "reload" },
        { role: "forceReload" },
        { type: "separator" },
        { role: "toggleDevTools" },
        { type: "separator" },
        { role: "resetZoom" },
        { role: "zoomIn" },
        { role: "zoomOut" },
        { type: "separator" },
        { role: "togglefullscreen" },
      ],
    },
    {
      label: "Help",
      submenu: [
        {
          label: "Check for Updates…",
          click: () => {
            try {
              autoUpdater.checkForUpdatesAndNotify();
            } catch (e) {
              dialog.showErrorBox("Update error", String(e));
            }
          },
        },
        {
          label: `About (v${app.getVersion()})`,
          click: () =>
            dialog.showMessageBox({
              type: "info",
              title: "About",
              message: `Crypto Intel Desktop\nVersion ${app.getVersion()}`,
            }),
        },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

// ----- Auto-updater wiring -----
function initAutoUpdater() {
  // Optional logging to file
  try {
    const log = require("electron-log");
    autoUpdater.logger = log;
    autoUpdater.logger.transports.file.level = "info";
  } catch {
    // logging is optional
  }

  autoUpdater.autoDownload = true;         // download updates automatically
  autoUpdater.autoInstallOnAppQuit = true; // install when user quits if they don't restart now

  autoUpdater.on("error", (err) => {
    console.warn("AutoUpdater error:", err?.message || err);
  });
  autoUpdater.on("update-available", (info) => {
    console.log("Update available:", info?.version);
  });
  autoUpdater.on("update-not-available", () => {
    console.log("No update available");
  });
  autoUpdater.on("update-downloaded", (info) => {
    const res = dialog.showMessageBoxSync({
      type: "question",
      buttons: ["Restart & Update", "Later"],
      defaultId: 0,
      cancelId: 1,
      title: "Update ready",
      message: "A new version of Crypto Intel has been downloaded.",
      detail: `Version ${info?.version} is ready to install. Restart now?`,
    });
    if (res === 0) {
      // Will quit the app and run the installer
      autoUpdater.quitAndInstall();
    }
  });

  // Initial check (a few seconds after ready to allow network to settle)
  setTimeout(() => {
    autoUpdater.checkForUpdatesAndNotify().catch(() => {});
  }, 5000);

  // Periodic checks (every 4 hours)
  setInterval(() => {
    autoUpdater.checkForUpdatesAndNotify().catch(() => {});
  }, 4 * 60 * 60 * 1000);
}

// ----- IPC (optional) -----
ipcMain.on("restart_app", () => {
  try {
    autoUpdater.quitAndInstall();
  } catch (_) {}
});

// ----- App lifecycle -----
app.whenReady().then(() => {
  buildMenu();
  createWindow();
  initAutoUpdater();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

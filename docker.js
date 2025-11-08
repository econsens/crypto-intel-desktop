// ==========================
// docker.js (clean + single spawn)
// ==========================
const { spawn } = require('child_process');

function run(cmd, args = [], opts = {}) {
  return new Promise((resolve) => {
    const child = spawn(cmd, args, { shell: true, windowsHide: true, ...opts });
    let stdout = '', stderr = '';
    child.stdout.on('data', d => stdout += d.toString());
    child.stderr.on('data', d => stderr += d.toString());
    child.on('close', code => resolve({ code, stdout: stdout.trim(), stderr: stderr.trim() }));
  });
}

async function dockerAvailable() {
  const res = await run('docker', ['version', '--format', '{{.Server.Version}}']);
  return res.code === 0;
}
async function containerRunning(name) {
  const res = await run('docker', ['ps', '--filter', `name=${name}`, '--format', '{{.Names}}']);
  if (res.code !== 0) return false;
  return res.stdout.split('\n').some(n => n.trim() === name);
}
async function containerExists(name) {
  const res = await run('docker', ['ps', '-a', '--filter', `name=${name}`, '--format', '{{.Names}}']);
  if (res.code !== 0) return false;
  return res.stdout.split('\n').some(n => n.trim() === name);
}
async function imageExists(image) {
  const res = await run('docker', ['image', 'inspect', image]);
  return res.code === 0;
}

/** Ensure the backend container is running (or create it). */
async function ensureContainerRunning({ container, image, port, dataDir }) {
  if (!(await dockerAvailable())) {
    return { ok: false, message: 'Docker is not available. Is Docker Desktop running?' };
  }
  if (await containerRunning(container)) return { ok: true, message: 'Container already running.' };

  if (await containerExists(container)) {
    const start = await run('docker', ['start', container]);
    if (start.code === 0) return { ok: true, message: 'Container started.' };
    return { ok: false, message: `Failed to start container: ${start.stderr || start.stdout}` };
  }

  if (!(await imageExists(image))) {
    return { ok: false, message: `Docker image "${image}" not found. Please build it first.` };
  }

  const volumeSpec = `${dataDir.replace(/\\/g, '/') }:/data`;
  const args = [
    'run','-d','--name', container,
    '-p', `127.0.0.1:${port}:8000`,
    '-v', volumeSpec,
    image,
    'uvicorn','app:app','--host','0.0.0.0','--port','8000'
  ];
  const create = await run('docker', args);
  if (create.code === 0) return { ok: true, message: 'Container created and started.' };
  return { ok: false, message: `Failed to run container: ${create.stderr || create.stdout}` };
}

/** Stop & remove the container (best-effort). */
async function stopContainer(container) {
  await run('docker', ['stop', container]);
  await run('docker', ['rm', container]);
  return { ok: true };
}

module.exports = { ensureContainerRunning, stopContainer };

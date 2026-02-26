import { app, BrowserWindow, ipcMain } from "electron";
import { execFile } from "node:child_process";
import path from "node:path";

const isDev = !app.isPackaged;

function createWindow() {
  const win = new BrowserWindow({
    width: 1100,
    height: 760,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  if (isDev) {
    win.loadURL("http://localhost:5173");
  } else {
    win.loadFile(path.join(__dirname, "../dist/index.html"));
  }
}

ipcMain.handle(
  "run-mlp",
  async (_event, payload: { csvPath: string; activation: string }) => {
    const binaryPath = path.join(process.cwd(), "build", "mlp_inference");

    return new Promise<{ rows: number; columns: number; outputs: number[] }>(
      (resolve, reject) => {
        execFile(
          binaryPath,
          ["--csv", payload.csvPath, "--activation", payload.activation],
          (err, stdout, stderr) => {
            if (err) {
              const stderrTrimmed = stderr.trim();
              if (stderrTrimmed.startsWith("{")) {
                try {
                  const parsed = JSON.parse(stderrTrimmed);
                  reject(
                    new Error(
                      parsed.error ?? "Falha ao executar a inferencia.",
                    ),
                  );
                  return;
                } catch {
                  reject(new Error(stderrTrimmed));
                  return;
                }
              }
              reject(new Error(stderrTrimmed || err.message));
              return;
            }

            try {
              const parsed = JSON.parse(stdout.trim());
              resolve(parsed);
            } catch {
              reject(new Error("Resposta invalida do backend C++."));
            }
          },
        );
      },
    );
  },
);

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

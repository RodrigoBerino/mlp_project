const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("mlpApi", {
  runMLP: (csvPath, activation) =>
    ipcRenderer.invoke("run-mlp", { csvPath, activation }),
});

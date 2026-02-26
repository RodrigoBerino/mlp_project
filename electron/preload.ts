import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("mlpApi", {
  runMLP: (csvPath: string, activation: "relu" | "sigmoid" | "tanh") =>
    ipcRenderer.invoke("run-mlp", { csvPath, activation }),
});

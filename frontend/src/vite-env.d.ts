/// <reference types="vite/client" />

declare global {
  interface Window {
    mlpApi: {
      runMLP: (
        csvPath: string,
        activation: "relu" | "sigmoid" | "tanh",
      ) => Promise<{ rows: number; columns: number; outputs: number[] }>;
    };
  }
}

export {};

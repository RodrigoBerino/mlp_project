import { useMemo, useState } from "react";

type Activation = "relu" | "sigmoid" | "tanh";

function App() {
  const [csvPath, setCsvPath] = useState("");
  const [activation, setActivation] = useState<Activation>("relu");
  const [rows, setRows] = useState<number | null>(null);
  const [columns, setColumns] = useState<number | null>(null);
  const [outputs, setOutputs] = useState<number[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const hasData = useMemo(
    () => rows !== null && columns !== null,
    [rows, columns],
  );

  const onDrop: React.DragEventHandler<HTMLDivElement> = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (!file) return;

    const droppedPath = (file as File & { path?: string }).path;
    if (!droppedPath?.toLowerCase().endsWith(".csv")) {
      setError("Por favor, solte um arquivo CSV válido.");
      return;
    }

    setCsvPath(droppedPath);
    setError("");
  };

  const runInference = async () => {
    if (!csvPath) {
      setError("Faça o drag-and-drop de um arquivo CSV antes de executar.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await window.mlpApi.runMLP(csvPath, activation);
      setRows(response.rows);
      setColumns(response.columns);
      setOutputs(response.outputs);
    } catch (err) {
      setRows(null);
      setColumns(null);
      setOutputs([]);
      setError(
        err instanceof Error ? err.message : "Erro inesperado na inferência.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container">
      <h3>MLP Desktop</h3>

      <section
        className="dropzone"
        onDrop={onDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        <p>Arraste e solte um arquivo CSV aqui</p>
      </section>

      <div className="controls">
        <label>
          Função de ativação:
          <select
            value={activation}
            onChange={(e) => setActivation(e.target.value as Activation)}
          >
            <option value="relu">ReLU</option>
            <option value="sigmoid">Sigmoid</option>
            <option value="tanh">Tanh</option>
          </select>
        </label>

        <button onClick={runInference} disabled={loading}>
          {loading ? "Executando..." : "Executar MLP"}
        </button>
      </div>

      {error && <p className="error">{error}</p>}

      {hasData && (
        <section className="results">
          <h2>Resultados</h2>
          <p>
            <strong>Dimensão dos dados:</strong> {rows} linhas x {columns}{" "}
            colunas
          </p>
          <p>
            <strong>Saída da inferência:</strong>
          </p>
          <ol>
            {outputs.map((output, index) => (
              <li key={index}>{output.toFixed(6)}</li>
            ))}
          </ol>
        </section>
      )}
    </main>
  );
}

export default App;

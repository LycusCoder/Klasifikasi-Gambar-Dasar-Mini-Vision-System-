import { useEffect, useMemo, useState } from "react";
import "./App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function formatPct(v) {
  if (v == null) return "-";
  return `${(v * 100).toFixed(1)}%`;
}

function StatusBadge({ status }) {
  const color = {
    IDLE: "bg-gray-600",
    TRAINING: "bg-yellow-600 animate-pulse",
    DONE: "bg-green-600",
    ERROR: "bg-red-600",
  }[status || "IDLE"];
  return (
    <span data-testid="training-status-badge" className={`inline-block px-3 py-1 rounded text-white text-sm ${color}`}>
      {status || "IDLE"}
    </span>
  );
}

function Canvas28({ data, scale = 6 }) {
  const [id] = useState(() => `c_${Math.random().toString(36).slice(2)}`);
  useEffect(() => {
    if (!data) return;
    const canvas = document.getElementById(id);
    const ctx = canvas.getContext("2d");
    const w = 28, h = 28;
    const imgData = ctx.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      const v = Math.max(0, Math.min(255, Math.round((data.flat ? data : data[Math.floor(i / w)][i % w]) * 255)));
      imgData.data[i * 4 + 0] = v;
      imgData.data[i * 4 + 1] = v;
      imgData.data[i * 4 + 2] = v;
      imgData.data[i * 4 + 3] = 255;
    }
    // draw scaled
    const tmp = document.createElement("canvas");
    tmp.width = w; tmp.height = h;
    tmp.getContext("2d").putImageData(imgData, 0, 0);
    canvas.width = w * scale; canvas.height = h * scale;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, w * scale, h * scale);
  }, [data, id, scale]);
  return <canvas id={id} width={28 * scale} height={28 * scale} className="border border-gray-700 rounded" />;
}

const Home = () => {
  const [status, setStatus] = useState("IDLE");
  const [message, setMessage] = useState("");
  const [metrics, setMetrics] = useState(null);
  const [epochs, setEpochs] = useState(5);
  const [batchSize, setBatchSize] = useState(64);
  const [samples, setSamples] = useState({ images: [], labels: [] });
  const [loadingTrain, setLoadingTrain] = useState(false);

  const pollStatus = async () => {
    try {
      const r = await axios.get(`${API}/train/status`);
      setStatus(r.data.status);
      setMessage(r.data.message || "");
      if (r.data.latest_metrics) setMetrics(r.data.latest_metrics);
    } catch (e) {
      console.error(e);
    }
  };

  const loadLatest = async () => {
    try {
      const r = await axios.get(`${API}/models/latest`);
      setMetrics(r.data);
    } catch {}
  };

  const loadSamples = async () => {
    try {
      const r = await axios.get(`${API}/samples?count=8`);
      setSamples(r.data);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    pollStatus();
    loadSamples();
    const t = setInterval(pollStatus, 2000);
    return () => clearInterval(t);
  }, []);

  const startTraining = async () => {
    try {
      setLoadingTrain(true);
      await axios.post(`${API}/train`, { epochs, batch_size: batchSize, model_name: "fashion_mnist_mlp" });
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingTrain(false);
    }
  };

  const history = metrics?.history || null;
  const testAcc = metrics?.test_accuracy ?? null;
  const testLoss = metrics?.test_loss ?? null;

  return (
    <div className="min-h-screen bg-black text-white">
      <div className="max-w-6xl mx-auto p-6">
        <h1 className="text-2xl font-semibold mb-2">Mini-Vision System: Fashion-MNIST MLP</h1>
        <div className="flex items-center gap-3 mb-6">
          <StatusBadge status={status} />
          <span data-testid="training-status-message" className="text-sm text-gray-400">{message}</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Left: Control */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 space-y-4" data-testid="model-control-panel">
            <h2 className="text-lg font-medium">Model Control &amp; Konfigurasi</h2>

            <div className="space-y-2">
              <label className="block text-sm text-gray-300">Epochs</label>
              <input data-testid="epochs-input" type="number" min={1} value={epochs} onChange={(e)=>setEpochs(parseInt(e.target.value||"1"))} className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 outline-none" />
            </div>

            <div className="space-y-2">
              <label className="block text-sm text-gray-300">Batch Size</label>
              <input data-testid="batch-size-input" type="number" min={1} value={batchSize} onChange={(e)=>setBatchSize(parseInt(e.target.value||"1"))} className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 outline-none" />
            </div>

            <div className="flex gap-3">
              <button data-testid="start-training-button" onClick={startTraining} disabled={status==="TRAINING" || loadingTrain} className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded disabled:opacity-50">
                {loadingTrain? "Starting..." : "Start Training & Export TFLite"}
              </button>
              <a data-testid="download-tflite-button" className={`px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded ${!metrics?"pointer-events-none opacity-50":""}`} href={`${API}/models/download`} target="_blank" rel="noreferrer">
                Download TFLite Model
              </a>
            </div>

            <div className="mt-4">
              <h3 className="text-sm text-gray-400 mb-2">Arsitektur Model (MLP)</h3>
              <pre className="bg-zinc-950 border border-zinc-800 rounded p-3 text-xs overflow-auto" data-testid="model-arch-code">
{`Sequential([
  Input(shape=(28,28)),
  Flatten(),
  Dense(512, activation='relu'),
  Dense(10, activation='softmax')
])`}
              </pre>
            </div>
          </div>

          {/* Right: Output */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 space-y-4" data-testid="output-panel">
            <h2 className="text-lg font-medium">Output &amp; Visualisasi</h2>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-zinc-950 border border-zinc-800 rounded p-4 text-center" data-testid="test-accuracy-card">
                <div className="text-xs text-gray-400">Akurasi (Test)</div>
                <div className="text-2xl font-semibold mt-2">{formatPct(testAcc)}</div>
              </div>
              <div className="bg-zinc-950 border border-zinc-800 rounded p-4 text-center" data-testid="test-loss-card">
                <div className="text-xs text-gray-400">Loss (Test)</div>
                <div className="text-2xl font-semibold mt-2">{testLoss?.toFixed ? testLoss.toFixed(3) : "-"}</div>
              </div>
            </div>

            <div>
              <h3 className="text-sm text-gray-300 mb-2">History per Epoch</h3>
              <div className="overflow-auto">
                <table className="w-full text-sm border border-zinc-800" data-testid="history-table">
                  <thead className="bg-zinc-800">
                    <tr>
                      <th className="px-2 py-1 text-left">Epoch</th>
                      <th className="px-2 py-1 text-left">acc</th>
                      <th className="px-2 py-1 text-left">val_acc</th>
                      <th className="px-2 py-1 text-left">loss</th>
                      <th className="px-2 py-1 text-left">val_loss</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history ? (
                      (history.accuracy || []).map((_, i) => (
                        <tr key={i} className="border-t border-zinc-800">
                          <td className="px-2 py-1">{i+1}</td>
                          <td className="px-2 py-1">{formatPct(history.accuracy?.[i] ?? null)}</td>
                          <td className="px-2 py-1">{formatPct(history.val_accuracy?.[i] ?? null)}</td>
                          <td className="px-2 py-1">{history.loss?.[i]?.toFixed?.(3) ?? "-"}</td>
                          <td className="px-2 py-1">{history.val_loss?.[i]?.toFixed?.(3) ?? "-"}</td>
                        </tr>
                      ))
                    ) : (
                      <tr><td className="px-2 py-2 text-gray-500" colSpan={5}>Belum ada history. Jalankan training.</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h3 className="text-sm text-gray-300 mb-2">Contoh Data (28x28)</h3>
              <div className="grid grid-cols-4 gap-3" data-testid="samples-grid">
                {samples.images.map((img, idx) => (
                  <div key={idx} className="flex flex-col items-center gap-1">
                    <Canvas28 data={img} />
                    <div className="text-xs text-gray-400">Label: {samples.labels[idx]}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-4">
              <h3 className="text-sm text-gray-400 mb-1">Langkah Selanjutnya</h3>
              <ul className="list-disc list-inside text-sm text-gray-300 space-y-1" data-testid="next-steps-list">
                <li>[ ] Add Inference Script</li>
                <li>[ ] Integrate TFLite Model into Mobile/Web App</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <Home />
    </div>
  );
}

export default App;
import React, { useEffect, useState } from "react";
import NetworkVisualizer from "./components/NetworkVisualizer";
import DrawingBoard from "./components/DrawingBoard";

function App() {
  const [weights, setWeights] = useState([]);
  const [epochs, setEpochs] = useState(10);
  const [lr, setLr] = useState(0.01);
  const [predInput, setPredInput] = useState(Array(784).fill(0));
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchWeights = async () => {
    const res = await fetch("/weights");
    const body = await res.json();
    setWeights(body.weights);
  };

  const train = async () => {
    await fetch("/train", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({epochs, lr})
    });
    await fetchWeights();
  };

  const predict = async () => {
    setLoading(true);
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: predInput }),
    });
    const body = await res.json();
    setPrediction(body.prediction);
    setLoading(false);
  };

  useEffect(() => { fetchWeights(); }, []);

  return (
    <div style={{padding: 20}}>
      <h1>MNIST Neural Network Visualizer</h1>
      <div>
        <label>Epochs: <input type="number" value={epochs} onChange={e => setEpochs(+e.target.value)} /></label>
        <label>Learning Rate: <input type="number" step="0.001" value={lr} onChange={e => setLr(+e.target.value)} /></label>
        <button onClick={train}>Train</button>
      </div>

      <div style={{marginTop: 20}}>
        <h2>Current Weights</h2>
        <NetworkVisualizer weights={weights} />
      </div>

      <div style={{ marginTop: 20 }}>
        <h2>Predict</h2>
        <DrawingBoard predInput={predInput} setPredInput={setPredInput} />
        <button onClick={predict} disabled={loading} style={{ marginTop: 10 }}>
          {loading ? "Predicting..." : "Run Prediction"}
        </button>
        {prediction !== null && (
          <p>
            Predicted digit: <b>{prediction}</b>
          </p>
        )}
      </div>
    </div>
  );
}

export default App;

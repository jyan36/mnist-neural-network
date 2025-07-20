import React from "react";

export default function NetworkVisualizer({ weights }) {
  return (
    <div>
      {weights.map((layer, i) => (
        <div key={i}>
          <h3>Layer {i+1} (&nbsp;{layer.length} × {layer[0].length}&nbsp;)</h3>
          <div style={{display: "grid", gridTemplateColumns: `repeat(${Math.min(layer[0].length, 20)}, 1fr)` }}>
            {layer.flat().slice(0,200).map((w, j) => (
              <div key={j} style={{
                width: 15, height: 15,
                background: `rgba(0,0,0,${Math.min(Math.max((w+1)/2,0),1)})`,
                margin: 1
              }} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

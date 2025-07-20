import React, { useRef, useEffect, useState } from "react";

const DrawingBoard = ({ predInput, setPredInput }) => {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const width = 280;  // 10px * 28 pixels
  const height = 280;
  const pixelSize = 10;

  const drawPixels = (ctx, pixels) => {
    ctx.clearRect(0, 0, width, height);
    for (let i = 0; i < 784; i++) {
      const val = pixels[i];
      const x = (i % 28) * pixelSize;
      const y = Math.floor(i / 28) * pixelSize;
      ctx.fillStyle = val ? "black" : "white";
      ctx.fillRect(x, y, pixelSize, pixelSize);
      ctx.strokeStyle = "lightgray";
      ctx.strokeRect(x, y, pixelSize, pixelSize);
    }
  };

  useEffect(() => {
    const ctx = canvasRef.current.getContext("2d");
    drawPixels(ctx, predInput);
  }, [predInput]);

  // Convert mouse coordinates to pixel index
  const getPixelIndex = (x, y) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const canvasX = x - rect.left;
    const canvasY = y - rect.top;
    const px = Math.floor(canvasX / pixelSize);
    const py = Math.floor(canvasY / pixelSize);
    return py * 28 + px;
  };

  const handlePointerDown = (e) => {
    setIsDrawing(true);
    handlePointerMove(e);
  };

  const handlePointerUp = () => {
    setIsDrawing(false);
  };

  const handlePointerMove = (e) => {
    if (!isDrawing) return;
    const index = getPixelIndex(e.clientX, e.clientY);
    if (index < 0 || index >= 784) return;
    setPredInput(prev => {
      if (prev[index] === 1) return prev; // already black
      const newPixels = [...prev];
      newPixels[index] = 1; // draw black
      return newPixels;
    });
  };

  // Clear the canvas/drawing
  const clearCanvas = () => {
    setPredInput(Array(784).fill(0));
  };

  return (
    <div style={{ border: "2px solid black", display: "inline-block", marginTop: 10 }}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{ cursor: "crosshair", backgroundColor: "white" }}
        onPointerDown={handlePointerDown}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
        onPointerMove={handlePointerMove}
      />
      <div style={{ marginTop: 10, textAlign: "center" }}>
        <button onClick={clearCanvas}>Clear</button>
      </div>
    </div>
  );
};

export default DrawingBoard;

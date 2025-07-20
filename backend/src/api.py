from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from src.load_mnist import load_data
from src.neural_network import NeuralNetwork
import threading
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

X_train, y_train, X_test, y_test = load_data()
nn = NeuralNetwork(input_size=784, hidden_sizes=[128, 64, 32], output_size=10)

class TrainRequest(BaseModel):
    epochs: int
    lr: float

class PredictRequest(BaseModel):
    input: list  # length 784

class WeightsResponse(BaseModel):
    weights: list  # nested lists

@app.on_event("startup")
def start_training():
    def train_loop():
        nn.train(X_train[:1000], y_train[:1000], epochs=10, lr=0.01, batch_size=32)
    threading.Thread(target=train_loop, daemon=True).start()

@app.post("/train")
def train(req: TrainRequest):
    nn.train(X_train[:1000], y_train[:1000], epochs=req.epochs, lr=req.lr, batch_size=32)
    return {"status": "trained"}

@app.post("/predict")
def predict(req: PredictRequest):
    x = np.array(req.input).reshape(1, -1)
    pred = nn.predict(x)[0]
    return {"prediction": int(pred)}

@app.get("/weights", response_model=WeightsResponse)
def get_weights():
    return {"weights": [w.tolist() for w in nn.weights]}

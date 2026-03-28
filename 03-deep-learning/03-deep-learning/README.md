# Stage 3 — Deep Learning

Building neural networks and deep learning models using PyTorch,
applied to real financial data.

## Notebooks

### 01 - Neural Network Fundamentals: Credit Risk
- Same German Credit dataset as XGBoost — direct comparison
- PyTorch neural network: 48 → 64 → 32 → 16 → 1 architecture
- 5,761 trainable parameters — all start random, all learned
- **Overfitting demonstrated visually** — training loss falls,
  validation loss rises — textbook divergence curve
- Early stopping at epoch 10 — restores best weights automatically
- **Result:** Neural Network AUC 0.8188 vs XGBoost 0.8056 — NN wins
- **Key concepts:** Forward pass, backpropagation, gradient descent,
  Adam optimiser, BCE loss, dropout, early stopping, batch training
- **Key lesson:** Neural networks need regularisation on small datasets.
  With 800 samples XGBoost is competitive — NN advantage grows with data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import math
import random

# ============================
# 1. Reproducibility
# ============================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ============================
# 2. Generate Synthetic Multivariate Time Series
# ============================
def generate_data(n_samples=1500):
    t = np.arange(n_samples)

    season1 = 10 * np.sin(2 * np.pi * t / 24)
    season2 = 5 * np.sin(2 * np.pi * t / 168)
    trend = 0.01 * t
    noise = np.random.normal(0, 1, n_samples)

    feature1 = season1 + trend + noise
    feature2 = season2 + noise
    feature3 = np.cos(2 * np.pi * t / 24)
    feature4 = np.sin(2 * np.pi * t / 48)
    feature5 = trend

    target = feature1 + 0.5 * feature2 + noise

    data = np.vstack([feature1, feature2, feature3, feature4, feature5]).T
    return data, target

data, target = generate_data()

# ============================
# 3. Preprocessing
# ============================
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

sequence_length = 48

def create_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(xs), np.array(ys)

X, y = create_sequences(data_scaled, target, sequence_length)

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# ============================
# 4. Baseline LSTM Model
# ============================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ============================
# 5. LSTM + Attention Model
# ============================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        attn_weights = torch.softmax(self.attn(lstm_output), dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context, attn_weights

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        output = self.fc(context)
        return output, attn_weights

# ============================
# 6. Training Function
# ============================
def train_model(model, train_loader, epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            output = model(xb)
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

# ============================
# 7. Evaluation Metrics
# ============================
def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        output = model(X_test)
        if isinstance(output, tuple):
            output = output[0]
        preds = output.numpy()
        y_true = y_test.numpy()

    mae = mean_absolute_error(y_true, preds)
    rmse = math.sqrt(mean_squared_error(y_true, preds))
    mape = np.mean(np.abs((y_true - preds) / y_true)) * 100
    return mae, rmse, mape, preds

# ============================
# 8. Train Baseline Model
# ============================
input_dim = X_train.shape[2]
hidden_dim = 64

baseline_model = LSTMModel(input_dim, hidden_dim)
print("Training Baseline LSTM Model")
train_model(baseline_model, train_loader)

baseline_mae, baseline_rmse, baseline_mape, baseline_preds = evaluate(
    baseline_model, X_test, y_test
)

# ============================
# 9. Train Attention Model
# ============================
attention_model = LSTMAttentionModel(input_dim, hidden_dim)
print("\nTraining LSTM + Attention Model")
train_model(attention_model, train_loader)

attn_mae, attn_rmse, attn_mape, attn_preds = evaluate(
    attention_model, X_test, y_test
)

# ============================
# 10. Results Comparison
# ============================
print("\nFinal Comparison:")
print(f"Baseline LSTM -> MAE: {baseline_mae:.4f}, RMSE: {baseline_rmse:.4f}, MAPE: {baseline_mape:.2f}%")
print(f"LSTM + Attention -> MAE: {attn_mae:.4f}, RMSE: {attn_rmse:.4f}, MAPE: {attn_mape:.2f}%")

# ============================
# 11. Attention Weight Visualization
# ============================
attention_model.eval()
with torch.no_grad():
    sample_output, sample_attn = attention_model(X_test[:1])

attn_weights = sample_attn.squeeze().numpy()

plt.figure(figsize=(10,4))
plt.plot(attn_weights)
plt.title("Attention Weights for Sample Forecast")
plt.xlabel("Time Step")
plt.ylabel("Attention Weight")
plt.show()
  

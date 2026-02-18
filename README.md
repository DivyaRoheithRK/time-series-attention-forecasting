# Advanced Time Series Forecasting with LSTM and Attention Mechanism

## 📌 Project Overview

This project implements an advanced multivariate time series forecasting system using deep learning models. The objective is to model complex temporal dependencies using an LSTM-based architecture enhanced with a self-attention mechanism and compare its performance against a baseline LSTM model.

The project demonstrates how attention mechanisms improve forecasting accuracy by allowing the model to focus on the most relevant time steps in the sequence.

---

## 📊 Dataset Description

A synthetic multivariate time series dataset was generated with:

- 5 correlated features
- Two seasonal components (daily and weekly patterns)
- A clear upward trend
- Gaussian noise for realism

The dataset simulates realistic forecasting scenarios commonly found in energy demand, traffic prediction, or financial time series.

Sequence length used: 48 time steps

---

## 🧠 Model Architecture

### 🔹 Baseline Model
- Single-layer LSTM
- Hidden dimension: 64
- Fully connected output layer

### 🔹 Attention-Enhanced Model
- LSTM layer
- Self-attention mechanism
- Context vector computation
- Final fully connected output layer

The attention mechanism assigns importance weights to different time steps, allowing the model to focus on relevant temporal patterns.

---

## 🔁 Cross-Validation Strategy

A time-series split was used, ensuring:
- Training is performed only on past data
- Testing is done on future data
- No data leakage occurs

This setup respects the temporal structure of forecasting problems.

---

## ⚙️ Hyperparameters

- Hidden size: 64
- Batch size: 32
- Learning rate: 0.001
- Epochs: 10
- Sequence length: 48
- Optimizer: Adam
- Loss function: Mean Squared Error (MSE)

---

## 📈 Evaluation Metrics

The models were evaluated using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

---

## 📊 Model Comparison

| Model | MAE | RMSE | MAPE |
|-------|------|------|------|
| LSTM Baseline | (Your Value) | (Your Value) | (Your Value)% |
| LSTM + Attention | (Your Value) | (Your Value) | (Your Value)% |

> Replace the placeholder values above with your actual results from running the code.

---

## 🔍 Attention Analysis

The attention weights indicate that the model assigns higher importance to recent time steps and specific seasonal intervals. This confirms that the attention mechanism effectively captures temporal dependencies and periodic patterns within the data.

---

## 🚀 How to Run

Install required packages:



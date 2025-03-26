# optiver-DNN-project
# Optiver Volatility Prediction with Deep Neural Network (DNN)

## Project Overview
This project aims to predict the realized volatility of each stock within each time window using deep neural networks (DNN). It is based on the Kaggle competition "Optiver Realized Volatility Prediction" and closely aligns with the BIOS 626 course (Machine Learning for Health Sciences) to practically master machine learning and deep learning fundamentals.

---

## Objectives
- Use structured financial time series data for modeling
- Build a multilayer perceptron (MLP) regression model to predict volatility
- Practice key concepts from BIOS 626 Weeks 6–11:
  - Single-layer and multi-layer neural networks
  - Backpropagation and gradient descent optimization
  - Regularization (Dropout, L2) to prevent overfitting
  - Model evaluation and generalization

---

## Project Structure
```
optiver-dnn-project/
├── data/                  # Raw parquet or aggregated CSV feature files
│   └── book_features.csv
├── models/                # Trained model checkpoints
│   └── dnn_model.pt
├── notebooks/             # Jupyter Notebook analysis files
│   └── optiver_dnn.ipynb
├── utils/                 # Utility scripts (feature engineering, metrics)
│   └── feature_engineering.py
├── requirements.txt       # Dependency list
├── README.md              # Project documentation
```

---

## Model Architecture
- Built using PyTorch with the following structure:
  - Input layer: 10–40 dimensions (aggregated features)
  - Hidden layers: 128 → 64 → 32
  - Activation function: ReLU
  - Output layer: 1 (predict continuous volatility)

- Loss function: MSELoss
- Optimizer: Adam + weight_decay (L2 regularization)
- Regularization techniques: Dropout + EarlyStopping

---

## Model Evaluation
- Metrics:
  - Mean Squared Error (MSE)
  - R² Score (coefficient of determination)
- Model selection techniques:
  - Holdout validation
  - 5-Fold Cross Validation
  - Grid Search (hyperparameter tuning)


## Future Work
- Experiment with more complex architectures (e.g., Transformer)
- Use LSTM or Temporal CNN for time-aware modeling
- Perform model interpretability analysis (SHAP, permutation importance)
- Explore model ensemble strategies for robustness

---

## Acknowledgements
- Kaggle: [Optiver Realized Volatility Prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction)
- BIOS 626: Machine Learning for Health Sciences

---

> Author: YOUR_NAME  
> Version: v1.0


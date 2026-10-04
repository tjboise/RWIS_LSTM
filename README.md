# RWIS-LSTM: Pavement Surface Grip Forecasting

This repository contains the code and data pipeline for the paper:

**"Forecasting Pavement Surface Grip in Winter Using Long Short-Term Memory Modeling"**  
Tianjie Zhang, Hao Wang — Rutgers University, Department of Civil and Environmental Engineering  
Submitted to *International Journal of Pavement Engineering (IJPE)*

---

## Overview

Pavement surface grip is a safety-critical variable during winter road conditions. This work proposes **RWIS-LSTM**, a domain-adapted LSTM architecture trained on Road Weather Information System (RWIS) sensor streams to forecast surface grip values up to 60 minutes ahead.

Key contributions:
- A **balancing algorithm** that corrects extreme distributional skewness in grip data, improving prediction of rare low-grip events
- **RWIS-LSTM**, which augments a standard LSTM by feeding the observed grip value as an additional input at each time step, improving generalization across unseen RWIS devices
- **Leave-One-Group-Out (LOGO) cross-validation** to evaluate the model on completely unseen devices, simulating real-world deployment
- Analysis of optimal input **sequence length** and multi-step **forecasting horizon** up to 60 minutes using an autoregressive approach

---

## Results

### Single-step prediction (next 10 minutes)

| Model | Test R² (mean) | Test R² (std) | ΔR² |
|-------|---------------|--------------|-----|
| **RWIS-LSTM** | **0.88** | **0.05** | **0.08** |
| LSTM | 0.84 | 0.08 | 0.13 |
| Stacked LSTM | 0.83 | 0.08 | 0.14 |
| GRU | 0.79 | 0.11 | 0.18 |
| XGBoost | 0.72 | 0.17 | 0.26 |
| Random Forest | 0.70 | 0.16 | 0.27 |

RWIS-LSTM achieves the highest mean test R² (0.88), lowest overfitting gap (ΔR² = 0.08), and lowest variance across folds (σ = 0.05), outperforming all baselines.

### Effect of balancing algorithm

The balancing algorithm reduces the Mean Squared Logarithmic Error (MSLE) on low-grip conditions while maintaining overall R² performance. MSLE for the unbalanced model is approximately 70% higher than for the balanced model.

### Optimal sequence length

Sequence lengths from 4 to 25 yield stable performance (R² ≈ 0.87). A window of **4 steps (40 minutes of history)** is identified as optimal, balancing accuracy and computational cost.

### Multi-step autoregressive forecasting (10–60 minutes)

Using an autoregressive strategy (predicted grip fed back at each step; weather features approximated via persistence):

| Horizon | R² | RMSE |
|---------|-----|------|
| 10 min | 0.74 | 0.030 |
| 20 min | ~0.56 | ~0.039 |
| 40 min | ~0.30 | ~0.050 |
| 60 min | ~0.17 | ~0.062 |

Near-term forecasts (10–20 minutes) retain the most operational value for maintenance decisions such as pre-emptive salt application.

### Baseline comparison (LOGO-CV, 3 seeds × 4 stations)

#### Overall mean metrics

| Model | Input | R² | RMSE | MAE | MAPE |
|-------|-------|----|------|-----|------|
| **RWIS-LSTM (HPO)** | x sequence + y history | **0.638** | **0.035** | **0.022** | **3.06%** |
| Random Forest | x (single step) | 0.378 | 0.042 | 0.030 | 4.19% |
| XGBoost | x (single step) | 0.200 | 0.046 | 0.038 | 5.15% |
| Decision Tree | x (single step) | 0.152 | 0.048 | 0.032 | 4.43% |
| Stacked LSTM (x-only, m2m) | x sequence | 0.132 | 0.053 | 0.032 | 4.53% |
| MLP | x (single step) | 0.015 | 0.056 | 0.040 | 5.39% |
| RNN (x-only, m2m) | x sequence | -0.023 | 0.060 | 0.041 | 5.58% |
| LSTM (x-only, m2m) | x sequence | -0.079 | 0.059 | 0.040 | 5.48% |
| GRU (x-only, m2m) | x sequence | -0.218 | 0.065 | 0.044 | 6.09% |

*m2m = many-to-many; evaluated on the last output step (y_{n+1}). All results: 3 seeds × 4 LOGO folds.*

#### Per-station R²

| Model | Station 158 | Station 26 | Station 27 | Station 29 |
|-------|-------------|------------|------------|------------|
| **RWIS-LSTM (HPO)** | **0.800** | **0.433** | **0.801** | **0.518** |
| Random Forest | 0.617 | -0.383 | 0.650 | 0.627 |
| XGBoost | 0.540 | -0.946 | 0.596 | 0.607 |
| Decision Tree | 0.537 | -1.024 | 0.554 | 0.542 |
| Stacked LSTM (x-only) | 0.375 | -0.558 | 0.496 | 0.213 |
| MLP | 0.609 | -0.677 | 0.408 | -0.282 |
| RNN (x-only) | 0.457 | -0.233 | 0.453 | -0.771 |
| LSTM (x-only) | 0.433 | -0.809 | 0.419 | -0.360 |
| GRU (x-only) | 0.413 | -0.673 | 0.430 | -1.044 |

Station 26 is the hardest fold for all models under LOGO cross-validation. RWIS-LSTM's use of y history as input is the key driver of its advantage over x-only baselines.

---

## Repository Structure

```
friction project/
├── generate training data.ipynb   # Data loading, cleaning, feature engineering
│                                  # → outputs preprocessed_rwis_data.xlsx
│
├── data analysis.ipynb            # Exploratory data analysis and Figure 1–2
├── plots.ipynb                    # Figure generation for the paper
│
├── multistep_realistic.ipynb      # Autoregressive multi-step prediction
│                                  # (k=1–6 steps, 10–60 min; no oracle features)
│
├── model 1/
│   ├── run1.ipynb                 # Full model comparison (all architectures, LOGO-CV)
│   ├── length2.ipynb              # Sequence-length sensitivity (WINDOW=4, seeds 42/43/44)
│   ├── seed42.ipynb               # Seed-42 RWIS-LSTM training
│   ├── seed43.ipynb               # Seed-43 RWIS-LSTM training (model 1 copy 2/)
│   ├── seed44.ipynb               # Seed-44 RWIS-LSTM training (model 1 copy/)
│   └── models/                   # Saved model weights (.h5)
│
├── model_try1/ model_nocopy/ ...  # Ablation / earlier experiments
│
└── paper/IJPE/                    # Manuscript, reviewer responses, figures
```

---

## Data

| Item | Detail |
|------|--------|
| Source | Utah Department of Transportation RWIS network |
| Period | November 1, 2021 – April 30, 2023 |
| Devices | 4 co-located stations along an Interstate highway corridor |
| Size | ~500,000 observations at 10-minute intervals |
| Features | 13 time-series variables (precipitation intensity, snow depth, solar radiation, surface temperature, surface status, surface water/ice/snow depth, air temperature, relative humidity, dew point, snowfall rate, total rain) |
| Target | Pavement surface grip value (continuous, 0–1) |

The raw data file `ParleysRWIS_20211101-20230430.xlsx` is not included in this repository. Please contact the authors for access.

---

## Requirements

```
python >= 3.9
tensorflow >= 2.10   (or tf-keras for TF 2.16+)
scikit-learn >= 1.0
scikeras >= 0.10
xgboost >= 1.7
scikit-optimize (skopt)
pandas >= 1.5
numpy >= 1.23
openpyxl >= 3.0
matplotlib >= 3.6
seaborn
```

Install all dependencies:

```bash
pip install tensorflow scikit-learn scikeras xgboost scikit-optimize \
            pandas numpy openpyxl matplotlib seaborn
```

---

## Reproducing Results

### Step 1 — Preprocess data

Place the raw data file in `friction project/`, then run:

```
friction project/generate training data.ipynb
```

This outputs `friction project/preprocessed_rwis_data.xlsx` (one-hot encoded, outlier-filtered, 10-minute resampled).

### Step 2 — Train and evaluate all models (LOGO-CV)

```
friction project/model 1/run1.ipynb
```

Trains XGBoost, Random Forest, Decision Tree, MLP, RNN, GRU, LSTM, Stacked LSTM, and RWIS-LSTM under LOGO cross-validation (seeds 42, 43, 44). Results saved to `model 1/*_results.xlsx`.

### Step 3 — Sequence length sensitivity

```
friction project/model 1/length2.ipynb
```

Evaluates RWIS-LSTM across input window sizes n = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25. Results saved to `model 1/LSTM_seed*_results.xlsx`.

### Step 4 — Multi-step autoregressive forecasting

```
friction project/multistep_realistic.ipynb
```

Trains single-step RWIS-LSTM (WINDOW=4) and evaluates autoregressive prediction for horizons of 10–60 minutes (k=1–6 steps) across seeds 42, 43, 44 and all LOGO folds. Results saved to `model 1/ar_60min_results.xlsx`.

### Step 5 — Regenerate figures

```
friction project/plots.ipynb
friction project/data analysis.ipynb
friction project/generate training data.ipynb   # for Figures 1, 2, 9
```

Figures are saved as high-resolution PNG files in `friction project/paper/IJPE/`.

---

## Model Architecture

RWIS-LSTM uses a many-to-one LSTM with input shape `(WINDOW+1, n_features+1)`:

- **Rows 0 to WINDOW-1**: historical feature vectors `[x_t, y_t]` (meteorological features + observed grip)
- **Row WINDOW**: persistence row `[x_{WINDOW}, y_{WINDOW-1}]` (latest observed features + last grip)
- **Output**: predicted grip at the next time step

```python
model = Sequential([
    LSTM(64, input_shape=(5, 19)),   # window+1=5, n_feat+1=19
    Dense(1)
])
model.compile(optimizer=Adam(1e-3), loss='mse')
```

Training uses early stopping (`patience=20`) on validation loss. Hyperparameter search (epochs ∈ {200, 300, 400}) is performed via 3-fold cross-validation within each LOGO training fold.

---

## Citation

```bibtex
@article{zhang2026rwislstm,
  title   = {Forecasting Pavement Surface Grip in Winter Using Long Short-Term Memory Modeling},
  author  = {Zhang, Tianjie and Wang, Hao},
  journal = {International Journal of Pavement Engineering},
  year    = {2026}
}
```

---

## Contact

Tianjie Zhang — [tjzhang37@gmail.com](mailto:tjzhang37@gmail.com)  
Hao Wang — [hwang.cee@rutgers.edu](mailto:hwang.cee@rutgers.edu)  
Department of Civil and Environmental Engineering, Rutgers University

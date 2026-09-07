# RWIS-LSTM: Pavement Surface Grip Forecasting

This repository contains the code for the paper:

**"Forecasting Pavement Surface Grip in Winter Using Long Short-Term Memory Modeling"**  
Tianjie Zhang, Hao Wang — Rutgers University, Department of Civil and Environmental Engineering

Submitted to *International Journal of Pavement Engineering (IJPE)*

---

## Project Structure

```
friction project/
├── generate training data.ipynb   # Data loading, cleaning, preprocessing
├── data analysis.ipynb            # EDA and data visualization
├── plots.ipynb                    # Figure generation for the paper
│
├── model 1/
│   ├── run1.ipynb                 # Main model comparison (all architectures, LOGO-CV)
│   ├── length2.ipynb              # RWIS-LSTM with WINDOW=4, seeds 42/43/44
│   └── seed42.ipynb               # Individual seed runs
│
├── model_try1/ model_nocopy/ ...  # Ablation / earlier experiments
│
└── multistep_realistic.ipynb      # NEW: Realistic multi-step prediction
                                   # (autoregressive + persistence, no oracle features)
```

---

## Key Methods

- **RWIS-LSTM**: Custom LSTM that feeds the previous grip value as an additional input at each time step, enabling direct memory of past road conditions
- **Leave-One-Group-Out (LOGO) CV**: Each of 4 RWIS devices is held out as the test set in turn, ensuring evaluation on completely unseen devices
- **Balancing Algorithm**: Downsamples over-represented high-grip bins (k=50 bins, max M=1000 per bin) to reduce bias toward dry-road conditions
- **Sequence length**: WINDOW=4 (40 minutes of history); shown to be optimal in Section 3.4

---

## Data

Raw data: `ParleysRWIS_20211101-20230430.xlsx` (not included — contact authors)  
- Source: Utah DOT, Interstate highway RWIS stations
- Period: November 2021 – April 2023
- 4 devices, ~500,000 observations at 10-minute intervals

---

## Requirements

```
tensorflow >= 2.10
scikit-learn
scikeras
xgboost
scikit-optimize (skopt)
pandas
numpy
openpyxl
matplotlib
seaborn
```

---

## Reproducing Results

1. Obtain the raw data file and place it in the `friction project/` directory
2. Run `generate training data.ipynb` to produce `preprocessed_rwis_data.xlsx`
3. Run `model 1/length2.ipynb` to train the RWIS-LSTM (seeds 42, 43, 44)
4. Run `multistep_realistic.ipynb` for the oracle vs. realistic multi-step comparison
5. Run `plots.ipynb` to regenerate all figures

---

## Citation

Zhang, T., & Wang, H. (2026). Forecasting Pavement Surface Grip in Winter Using Long Short-Term Memory Modeling. *International Journal of Pavement Engineering*.

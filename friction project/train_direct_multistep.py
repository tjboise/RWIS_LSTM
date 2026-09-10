"""
Direct multi-step grip prediction — train one model per (fold, k).
k=1 result reused from autoregressive experiment.
k=2,4,8: train new models with target = y_{t+k}, seed=42.
Input: (5, 19) — 4 history rows + persistence row (no oracle).
"""
import os, random, warnings, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras
from tf_keras import Sequential
from tf_keras.layers import LSTM, Dense
from tf_keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

BASE_DIR   = r'C:\Users\Tianjie Zhang\OneDrive - Rutgers University\friction_project\friction project'
DATA_PATH  = os.path.join(BASE_DIR, 'preprocessed_rwis_data.xlsx')
MODEL_DIR  = os.path.join(BASE_DIR, 'model 1', 'models_direct')
AR_RESULTS = os.path.join(BASE_DIR, 'model 1', 'multistep_realistic_results.xlsx')
OUT_PATH   = os.path.join(BASE_DIR, 'model 1', 'direct_multistep_results.xlsx')

WINDOW         = 4
SEED           = 42
HORIZONS_TRAIN = [2, 4, 8]   # k=1 already have from autoregressive experiment
ALL_HORIZONS   = [1, 2, 4, 8]

os.makedirs(MODEL_DIR, exist_ok=True)

# reproducibility
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)

# ── data ─────────────────────────────────────────────────────────────────────
print('Loading data...')
df = pd.read_excel(DATA_PATH, engine='openpyxl')
label_col    = 'SurfaceGrip'
feature_cols = [c for c in df.columns
                if c not in ['SampleTime', 'RwisControllerIntId', 'time_diff', label_col]]
n_feat = len(feature_cols)
print(f'  {len(df)} rows | {n_feat} features | devices: {sorted(df["RwisControllerIntId"].unique())}')

# ── sequence builder (direct k-step, no oracle) ───────────────────────────────
def build_direct_sequences(df_device, feature_cols, label_col, window, k):
    """
    Input : (window+1, n_feat+1) — 4 history rows + persistence row
    Target: y_{t+k}  (k steps ahead)
    """
    grp = df_device.reset_index(drop=True)
    X, y = [], []
    for i in range(window, len(grp) - k + 1):
        hist    = grp.iloc[i - window:i]
        feat    = hist[feature_cols].values          # (window, n_feat)
        hy      = hist[[label_col]].values            # (window, 1)
        seq_hist = np.hstack([feat, hy])              # (window, n_feat+1)
        # persistence: repeat last observed features (no oracle)
        persist_feat = feat[-1:, :]                   # (1, n_feat)
        curr_row     = np.hstack([persist_feat, hy[-1:]])  # (1, n_feat+1)
        seq          = np.vstack([seq_hist, curr_row])     # (5, n_feat+1)
        X.append(seq)
        y.append(grp.iloc[i + k - 1][label_col])     # target k steps ahead
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ── balancing ─────────────────────────────────────────────────────────────────
def balance(X, y, bins=50, max_per_bin=1000, rng=42):
    np.random.seed(rng)
    edges = np.linspace(y.min(), y.max(), bins + 1)
    idx   = np.digitize(y, edges, right=True)
    sel   = []
    for b in range(1, bins + 1):
        ids = np.where(idx == b)[0]
        if len(ids) > max_per_bin:
            ids = np.random.choice(ids, max_per_bin, replace=False)
        sel.extend(ids.tolist())
    sel = np.array(sel)
    return X[sel], y[sel]

# ── model factory ─────────────────────────────────────────────────────────────
def build_lstm(input_shape):
    m = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(1)
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    return m

# ── metrics ───────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    mse  = float(mean_squared_error(y_true, y_pred))
    return dict(
        MSE  = mse,
        RMSE = float(np.sqrt(mse)),
        MAE  = float(mean_absolute_error(y_true, y_pred)),
        R2   = float(r2_score(y_true, y_pred))
    )

# ── LOGO fold ordering (must match training) ──────────────────────────────────
logo = LeaveOneGroupOut()
all_X, all_y, all_g = [], [], []
for gid, grp in df.groupby('RwisControllerIntId'):
    for i in range(WINDOW, len(grp)):
        all_g.append(gid)
all_g  = np.array(all_g)
# just need the group array for split ordering
dummy  = np.zeros(len(all_g))

fold_station = {}
for fold, (_, te_idx) in enumerate(logo.split(dummy, dummy, all_g), start=1):
    fold_station[fold] = all_g[te_idx][0]
print('Fold -> station:', fold_station)

# ── training loop ─────────────────────────────────────────────────────────────
rows = []

for k in HORIZONS_TRAIN:
    minutes = k * 10
    print(f'\n===== k={k} ({minutes} min) =====')

    # build full dataset for this horizon
    all_X_k, all_y_k, all_g_k = [], [], []
    for gid, grp in df.groupby('RwisControllerIntId'):
        Xi, yi = build_direct_sequences(grp, feature_cols, label_col, WINDOW, k)
        all_X_k.append(Xi); all_y_k.append(yi)
        all_g_k.extend([gid] * len(yi))
    all_X_k = np.concatenate(all_X_k)
    all_y_k = np.concatenate(all_y_k)
    all_g_k = np.array(all_g_k)
    print(f'  Dataset shape: {all_X_k.shape}')

    for fold, (tr_idx, te_idx) in enumerate(
            logo.split(all_X_k, all_y_k, all_g_k), start=1):
        station = all_g_k[te_idx][0]
        t0 = time.time()
        print(f'  fold {fold} / station {station} ...', end='', flush=True)

        X_tr, y_tr = all_X_k[tr_idx], all_y_k[tr_idx]
        X_te, y_te = all_X_k[te_idx], all_y_k[te_idx]

        # balance training set
        set_seed(SEED)
        X_tr_b, y_tr_b = balance(X_tr, y_tr, rng=SEED)

        # validation split (20%)
        n_val = int(len(X_tr_b) * 0.2)
        X_val, y_val = X_tr_b[:n_val], y_tr_b[:n_val]
        X_trn, y_trn = X_tr_b[n_val:], y_tr_b[n_val:]

        # train
        model = build_lstm(X_trn.shape[1:])
        model.fit(
            X_trn, y_trn,
            validation_data=(X_val, y_val),
            epochs=300,
            batch_size=32,
            callbacks=[EarlyStopping(monitor='val_loss', patience=20,
                                     restore_best_weights=True)],
            verbose=0
        )

        # save model
        model_path = os.path.join(MODEL_DIR, f'direct_k{k}_seed{SEED}_fold{fold}.h5')
        model.save(model_path)

        # evaluate
        y_pred = model.predict(X_te, verbose=0).ravel()
        m = metrics(y_te, y_pred)
        rows.append(dict(seed=SEED, fold=fold, station=station,
                         k=k, minutes=minutes, approach='direct', **m))
        dt = time.time() - t0
        print(f' R2={m["R2"]:.3f} RMSE={m["RMSE"]:.4f}  [{dt:.0f}s]')

# ── merge with k=1 autoregressive result ──────────────────────────────────────
print('\nLoading k=1 autoregressive results...')
ar_df = pd.read_excel(AR_RESULTS)
k1_ar = ar_df[(ar_df['k'] == 1) & (ar_df['approach'] == 'autoregressive') & (ar_df['seed'] == SEED)].copy()
k1_ar['approach'] = 'direct'   # k=1 direct == k=1 autoregressive
rows_df = pd.DataFrame(rows)
combined = pd.concat([k1_ar[['seed','fold','station','k','minutes','approach','MSE','RMSE','MAE','R2']],
                      rows_df], ignore_index=True)
combined.to_excel(OUT_PATH, index=False)
print(f'Saved -> {OUT_PATH}')

# ── summary ───────────────────────────────────────────────────────────────────
summary = (combined.groupby(['minutes','approach'])[['R2','RMSE','MAE']]
           .agg(['mean','std']).round(4))
summary.columns = ['_'.join(c) for c in summary.columns]
print('\n=== Direct Multi-Step Summary ===')
print(summary.to_string())

# also load oracle + autoregressive for comparison
oracle_ar = ar_df[ar_df['seed'] == SEED].copy()
full = pd.concat([oracle_ar, combined], ignore_index=True)
full_summary = (full.groupby(['minutes','approach'])[['R2','RMSE']]
                .mean().unstack('approach').round(4))
print('\n=== Full Comparison (seed=42 mean across folds) ===')
print(full_summary.to_string())

with pd.ExcelWriter(OUT_PATH, engine='openpyxl', mode='a',
                    if_sheet_exists='replace') as writer:
    summary.to_excel(writer, sheet_name='Summary')
    full_summary.to_excel(writer, sheet_name='Comparison')
print('Done.')

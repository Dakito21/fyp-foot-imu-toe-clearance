import glob
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

OUT_SUM = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'imu_mocap_comparison.csv'))

per_stride_files = sorted(glob.glob(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'phase1_9_per_stride_*.csv'))))
print('Found per-stride files:', [os.path.basename(p) for p in per_stride_files])

rows = []
for f in per_stride_files:
    part = os.path.basename(f).replace('phase1_9_per_stride_','').replace('.csv','')
    df = pd.read_csv(f)
    # ensure numeric
    for c in ['mtc_imu_mm','mtc_mocap_mm','err_mm','dt_tc_ms','dt_ic_ms']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    mask = df['mtc_imu_mm'].notna() & df['mtc_mocap_mm'].notna()
    dfm = df[mask].copy()
    if dfm.shape[0] == 0:
        continue
    X = dfm['mtc_imu_mm'].to_numpy().reshape(-1,1)
    y = dfm['mtc_mocap_mm'].to_numpy()
    orig_err = (dfm['mtc_imu_mm'] - dfm['mtc_mocap_mm']).to_numpy()
    def stats(e):
        return len(e), float(np.mean(e)), float(np.median(e)), float(np.sqrt(np.mean(e**2)))
    raw_n, raw_mean, raw_med, raw_rmse = stats(orig_err)
    # candidate 1: per-stride-ground (median delta)
    delta = np.median(dfm['mtc_mocap_mm'] - dfm['mtc_imu_mm'])
    corrected_ps = dfm['mtc_imu_mm'] + delta
    err_ps = corrected_ps - dfm['mtc_mocap_mm']
    ps_stats = stats(err_ps.to_numpy())
    # candidate 2: OLS on strides with dt_ic_ms within 100ms of median
    # compute median dt_ic and filter
    if 'dt_ic_ms' in dfm.columns:
        dt_med = np.median(dfm['dt_ic_ms'].dropna())
        dt_mask = np.abs(dfm['dt_ic_ms'] - dt_med) <= 100.0
    else:
        dt_mask = np.ones(len(dfm), dtype=bool)
    if dt_mask.sum() < 3:
        dt_mask = np.ones(len(dfm), dtype=bool)
    Xf = X[dt_mask]
    yf = y[dt_mask]
    ols = LinearRegression().fit(Xf, yf)
    pred_ols = ols.predict(X)
    err_ols = pred_ols - y
    ols_stats = stats(err_ols)
    # candidate 3: RANSAC
    try:
        ransac = RANSACRegressor(LinearRegression(), min_samples=max(3, int(0.5*len(Xf))))
        ransac.fit(Xf, yf)
        pred_ran = ransac.predict(X)
        err_ran = pred_ran - y
        ran_stats = stats(err_ran)
    except Exception:
        ran_stats = (len(y), np.nan, np.nan, np.nan)
    # candidate 4: OLS after removing large err outliers
    err0 = orig_err
    keep = np.abs(err0) < (2*np.std(err0))
    if keep.sum() >= 3:
        ols2 = LinearRegression().fit(X[keep], y[keep])
        pred_ols2 = ols2.predict(X)
        err_ols2 = pred_ols2 - y
        ols2_stats = stats(err_ols2)
    else:
        ols2_stats = (len(y), np.nan, np.nan, np.nan)
    # choose best by RMSE
    candidates = {
        'raw': (raw_n, raw_mean, raw_med, raw_rmse),
        'per_stride_ground': ps_stats,
        'ols': ols_stats,
        'ransac': ran_stats,
        'ols_outlier_removed': ols2_stats
    }
    # prefer numeric RMSE
    best = None
    best_rmse = np.inf
    for k,v in candidates.items():
        if np.isfinite(v[3]) and v[3] < best_rmse:
            best = k; best_rmse = v[3]
    # compute corrected values based on best
    if best == 'per_stride_ground':
        dfm['mtc_imu_mm_corrected'] = dfm['mtc_imu_mm'] + delta
    elif best == 'ols':
        dfm['mtc_imu_mm_corrected'] = ols.predict(dfm['mtc_imu_mm'].to_numpy().reshape(-1,1))
    elif best == 'ransac':
        dfm['mtc_imu_mm_corrected'] = ransac.predict(dfm['mtc_imu_mm'].to_numpy().reshape(-1,1))
    elif best == 'ols_outlier_removed':
        dfm['mtc_imu_mm_corrected'] = ols2.predict(dfm['mtc_imu_mm'].to_numpy().reshape(-1,1))
    else:
        dfm['mtc_imu_mm_corrected'] = dfm['mtc_imu_mm']
    dfm['err_mm_corrected'] = dfm['mtc_imu_mm_corrected'] - dfm['mtc_mocap_mm']
    corr_n, corr_mean, corr_med, corr_rmse = stats(dfm['err_mm_corrected'].to_numpy())
    out_final = os.path.abspath(os.path.join(os.path.dirname(f), os.path.basename(f).replace('.csv','_final.csv')))
    dfm.to_csv(out_final, index=False)
    rows.append({
        'participant': part,
        'N': raw_n,
        'bias_raw_mm': raw_mean,
        'rmse_raw_mm': raw_rmse,
        'bias_corrected_mm': corr_mean,
        'rmse_corrected_mm': corr_rmse,
        'method_used': best,
        'corrected_file': out_final
    })

# write summary
if rows:
    pd.DataFrame(rows).to_csv(OUT_SUM, index=False)
    print('Wrote summary to', OUT_SUM)
else:
    print('No per-stride files processed')

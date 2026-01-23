import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

fn = 'phase1_9_per_stride_4d91.csv'
df = pd.read_csv(fn)
X = df['mtc_imu_mm'].to_numpy().reshape(-1,1)
y = df['mtc_mocap_mm'].to_numpy()

def stats(err):
    n = len(err)
    mean = np.mean(err)
    med = np.median(err)
    rmse = np.sqrt(np.mean(err**2))
    return n, mean, med, rmse

orig_err = df['mtc_imu_mm'].to_numpy() - df['mtc_mocap_mm'].to_numpy()
print('Original:', stats(orig_err))

# per-stride ground (median delta)
delta = np.median(df['mtc_mocap_mm'] - df['mtc_imu_mm'])
corrected_ps = df['mtc_imu_mm'] + delta
err_ps = corrected_ps.to_numpy() - df['mtc_mocap_mm'].to_numpy()
print('Per-stride-ground delta mm =', delta)
print('Per-stride-ground:', stats(err_ps))

# OLS regression
lr = LinearRegression()
lr.fit(X, y)
print('OLS coef:', lr.coef_[0], 'intercept:', lr.intercept_)
pred_lr = lr.predict(X)
err_lr = pred_lr - y
print('OLS:', stats(err_lr))

# RANSAC robust regression
ransac = RANSACRegressor(LinearRegression(), min_samples=max(3, int(0.5*len(X))))
ransac.fit(X, y)
coef = ransac.estimator_.coef_[0]
inter = ransac.estimator_.intercept_
print('RANSAC coef:', coef, 'intercept:', inter)
pred_ransac = ransac.predict(X)
err_ransac = pred_ransac - y
print('RANSAC:', stats(err_ransac))

# Outlier removal + OLS
err0 = orig_err
mask = np.abs(err0) < (2*np.std(err0))
print('Outlier removal kept', mask.sum(), 'of', len(mask))
lr2 = LinearRegression()
lr2.fit(X[mask], y[mask])
pred_lr2 = lr2.predict(X)
err_lr2 = pred_lr2 - y
print('OLS after outlier removal:', stats(err_lr2))

# Choose best
methods = {
    'orig': stats(orig_err),
    'per_stride_ground': stats(err_ps),
    'ols': stats(err_lr),
    'ransac': stats(err_ransac),
    'ols_outlier_removed': stats(err_lr2)
}

print('\nSummary (n, mean, median, rmse):')
for k,v in methods.items():
    print(k, v)

# Save corrected per-stride CSV for best method (lowest rmse)
best = min(methods.items(), key=lambda kv: kv[1][3])[0]
print('\nBest method:', best)
if best == 'per_stride_ground':
    df['mtc_imu_mm_corrected'] = corrected_ps
elif best == 'ols':
    df['mtc_imu_mm_corrected'] = pred_lr
elif best == 'ransac':
    df['mtc_imu_mm_corrected'] = pred_ransac
elif best == 'ols_outlier_removed':
    df['mtc_imu_mm_corrected'] = pred_lr2
else:
    df['mtc_imu_mm_corrected'] = df['mtc_imu_mm']

df['err_mm_corrected'] = df['mtc_imu_mm_corrected'] - df['mtc_mocap_mm']
df.to_csv('phase1_9_per_stride_4d91_corrected.csv', index=False)
print('Wrote phase1_9_per_stride_4d91_corrected.csv')

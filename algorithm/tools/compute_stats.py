import sys
import pandas as pd
import numpy as np

if len(sys.argv) < 2:
    print('Usage: python compute_stats.py <csv-file>')
    sys.exit(2)

fn = sys.argv[1]
df = pd.read_csv(fn)
cols = ['err_mm','dt_tc_ms','dt_ic_ms']
for c in cols:
    s = pd.to_numeric(df[c], errors='coerce').dropna()
    n = len(s)
    mean = s.mean()
    med = s.median()
    rmse = np.sqrt((s**2).mean())
    print(f"{c}: N={n}, mean={mean:.3f}, median={med:.3f}, rmse={rmse:.3f}")

import csv
import os
import subprocess
import sys

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CSV = os.path.join(BASE, 'phase1_9_summary.csv')
PY = sys.executable
SCRIPT = os.path.join(BASE, 'MFC_dataset_testing.py')

if not os.path.exists(CSV):
    print('Summary CSV not found:', CSV); sys.exit(1)

rows = []
with open(CSV, newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append(row)

# selection: final_rmse_mm > 30 OR matched_rmse_mm>40 OR pos_tc_rmse_mm>30
sel = []
for r in rows:
    try:
        fr = float(r.get('final_rmse_mm') or 0)
    except:
        fr = 0
    try:
        mr = float(r.get('matched_rmse_mm') or 0)
    except:
        mr = 0
    try:
        pr = float(r.get('pos_tc_rmse_mm') or 0)
    except:
        pr = 0
    if fr > 30 or mr > 40 or pr > 30:
        sel.append(r['participant'])

print('Selected participants for per-stride export:', sel)

for p in sel:
    outname = os.path.join(BASE, f'phase1_9_per_stride_{p}.csv')
    cmd = [PY, SCRIPT, '--data_folder', os.path.join(BASE, 'sensorpositoncomparison', 'sensorpositoncomparison-v1.0.0-beta'), '--participant', p, '--test', 'normal_10', '--side', 'left', '--validate_by', 'auto', '--tc_threshold_ms', '100', '--min_matched_pairs', '5', '--max_xcorr_lag_s', '2.0', '--vertical_align', 'per_stride_ground', '--export_per_stride', outname]
    print('Running', p)
    subprocess.run(cmd)

print('Done')

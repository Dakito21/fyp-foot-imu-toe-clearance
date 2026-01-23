import os
import re
import csv
import subprocess
import sys

DATA_ROOT = sys.argv[1] if len(sys.argv) > 1 else r"path/to/sensorpositoncomparison-v1.0.0-beta\data"
PY = sys.executable
SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MFC_dataset_testing.py'))
OUT_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'phase1_9_summary.csv'))

participants = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
print('Found participants:', participants)

rows = []
for p in participants:
    print('\nRunning participant', p)
    cmd = [PY, SCRIPT, '--data_folder', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sensorpositoncomparison', 'sensorpositoncomparison-v1.0.0-beta')), '--participant', p, '--test', 'normal_10', '--side', 'left', '--validate_by', 'auto', '--tc_threshold_ms', '100', '--min_matched_pairs', '5', '--max_xcorr_lag_s', '2.0', '--vertical_align', 'per_stride_ground']
    print('CMD:', ' '.join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + '\n' + proc.stderr

    # regex extractors
    matched_re = re.search(r"\[Phase 1\.9 \(mocap-events\)\] Bias=([-0-9\.]+) mm \| RMSE=([-0-9\.]+) mm \| N=(\d+)", out)
    pos_tc_re = re.search(r"\[Phase 1\.9 Position @TC\] N=(\d+) \| Bias=([-0-9\.]+) mm \| RMSE=([-0-9\.]+) mm", out)
    final_re = re.search(r"\[Phase 1\.9\] Bias=([-0-9\.]+) mm \| RMSE=([-0-9\.]+) mm \| N=(\d+)", out)

    matched_bias = matched_rmse = matched_n = ''
    pos_tc_n = pos_tc_bias = pos_tc_rmse = ''
    final_bias = final_rmse = final_n = ''

    if matched_re:
        matched_bias = float(matched_re.group(1))
        matched_rmse = float(matched_re.group(2))
        matched_n = int(matched_re.group(3))
    if pos_tc_re:
        pos_tc_n = int(pos_tc_re.group(1))
        pos_tc_bias = float(pos_tc_re.group(2))
        pos_tc_rmse = float(pos_tc_re.group(3))
    if final_re:
        final_bias = float(final_re.group(1))
        final_rmse = float(final_re.group(2))
        final_n = int(final_re.group(3))

    rows.append({
        'participant': p,
        'matched_bias_mm': matched_bias,
        'matched_rmse_mm': matched_rmse,
        'matched_N': matched_n,
        'pos_tc_bias_mm': pos_tc_bias,
        'pos_tc_rmse_mm': pos_tc_rmse,
        'pos_tc_N': pos_tc_n,
        'final_bias_mm': final_bias,
        'final_rmse_mm': final_rmse,
        'final_N': final_n
    })

# write CSV
with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print('\nWrote summary to', OUT_CSV)

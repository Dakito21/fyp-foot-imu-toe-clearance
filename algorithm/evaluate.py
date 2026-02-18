
"""evaluate.py

Evaluation stage (frozen params):
- Load toe offsets from a calibration CSV
- Run the full reporting pipeline per row
- Write a summary CSV

Example:
python evaluate.py \
  --data_folder ".../sensorpositioncomparison-v1.0.0-beta" \
  --toe_offset_csv toe_offset_calibration_xz.csv \
  --event_source ml --event_model_pt param_stageA.pt \
  --out_csv imu_eval_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

import pipeline_core as core


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_folder', required=True)
    ap.add_argument('--toe_offset_csv', required=True)
    ap.add_argument('--out_csv', default='imu_eval_summary.csv')

    ap.add_argument('--padding_s', type=float, default=3.0)
    ap.add_argument('--gate_s', type=float, default=0.15)

    ap.add_argument('--event_source', default='imu', choices=['imu', 'ml'])
    ap.add_argument('--event_model_pt', default=None)
    ap.add_argument('--no_dynamic_zupt', action='store_true')
    ap.add_argument('--no_dynamic_gate', action='store_true')

    args = ap.parse_args()

    df = pd.read_csv(args.toe_offset_csv)

    # If the CSV only has 1 row, replicate it for all tests of that participant
    if len(df) == 1:
        p = str(df.loc[0, "participant"])
        side = str(df.loc[0, "side"]).lower()

        # get all tests available for that participant from the dataset index
        dataset = core.legacy.SensorPositionComparison2019Mocap(
            memory=core.legacy.Memory("./cache"),
            data_folder=args.data_folder,
            data_padding_s=float(args.padding_s),
        )
        idx = dataset.create_index()   # columns: participant, test
        tests = idx[idx["participant"].astype(str) == p]["test"].astype(str).unique().tolist()

        # replicate the single row for each test
        base = df.iloc[0].to_dict()
        rows = []
        for t in tests:
            r = dict(base)
            r["test"] = t
            r["side"] = side
            rows.append(r)

        df = pd.DataFrame(rows)

    required = {'participant', 'test', 'side', 'toe_x_cm', 'toe_z_cm'}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f'toe_offset_csv missing columns: {sorted(missing)}')

    # --- Fallback map: use normal_10 toe offset for all tests (same participant+side) ---
    df["participant"] = df["participant"].astype(str)
    df["test"] = df["test"].astype(str)
    df["side"] = df["side"].astype(str).str.lower()

    normal_df = df[df["test"] == "normal_10"].copy()

    if normal_df.empty:
        raise SystemExit("Need at least one row with test=normal_10 in toe_offset_csv to use as fallback.")

    # map (participant, side) -> (toe_x_cm, toe_z_cm) from normal_10
    normal_toe_map = {
        (str(r["participant"]), str(r["side"]).lower()): (float(r["toe_x_cm"]), float(r["toe_z_cm"]))
        for _, r in normal_df.iterrows()
    }
    # Pipeline is run in fully-dynamic mode
    toggles = core.PipelineToggles(dynamic_cutoff=True, dynamic_zupt=True, dynamic_ic_refine=True, dynamic_gate=True)

    eventnet_model = None
    eventnet_meta = None
    if args.event_source == 'ml':
        if not args.event_model_pt:
            raise SystemExit('--event_source ml requires --event_model_pt')
        eventnet_model, eventnet_meta = core.legacy.load_eventnet(args.event_model_pt, device="cpu")


    rows = []
    for _, r in df.iterrows():
        p = str(r['participant'])
        t = str(r['test'])
        s = str(r['side']).lower()
        # Always use normal_10 offsets for this participant+side
        if (p, s) not in normal_toe_map:
            raise KeyError(f"No normal_10 toe offset found for participant={p} side={s}")
        toe_x_cm, toe_z_cm = normal_toe_map[(p, s)]


        sensor = r['imu_sensor'] if 'imu_sensor' in df.columns and pd.notna(r.get('imu_sensor')) else None
        toe_marker = r['toe_marker'] if 'toe_marker' in df.columns and pd.notna(r.get('toe_marker')) else None

        try:
            trial = core.load_trial(
                data_folder=args.data_folder,
                participant=p,
                test=t,
                side=s,
                padding_s=args.padding_s,
                sensor=sensor,
                toe_marker=toe_marker,
            )

            # evaluation uses the legacy row reporter for now (already produces your desired flat row)
            row = core.legacy.run_pipeline_for_row_report(
                data_folder=args.data_folder,
                participant=p,
                test=t,
                side=s,
                toe_x_cm=toe_x_cm,
                toe_z_cm=toe_z_cm,
                sensor=trial.sensor,
                toe_marker=trial.toe_marker,
                padding_s=args.padding_s,
                event_source=args.event_source,
                eventnet_model=eventnet_model,
                eventnet_meta=eventnet_meta,
                quiet=True,
                print_reject_counts=False,
                gate_s=args.gate_s,
                dynamic_gate=toggles.dynamic_gate,
                dynamic_zupt=toggles.dynamic_zupt,
                dynamic_ic_refine=toggles.dynamic_ic_refine,
                dynamic_cutoff=toggles.dynamic_cutoff,
            )
            row['event_source'] = args.event_source
            rows.append(row)

        except Exception as e:
            rows.append({
                'participant': p,
                'test': t,
                'side': s,
                'toe_x_cm': toe_x_cm,
                'toe_z_cm': toe_z_cm,
                'error': str(e),
                'event_source': args.event_source,
            })

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_df.to_csv(out_path, index=False)
    print(f'[EVAL] Saved: {out_path.resolve()}')
    print(out_df.head(15).to_string(index=False))


if __name__ == '__main__':
    main()




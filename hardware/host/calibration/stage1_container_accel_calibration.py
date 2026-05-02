#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["seq","t_ms","ax_g","ay_g","az_g","gx_dps","gy_dps","gz_dps"]
ACCEL_COLS = ["ax_g","ay_g","az_g"]

def compact_ranges(values: Iterable[int]) -> str:
    values = sorted(set(int(v) for v in values))
    if not values: return "none"
    ranges=[]; start=prev=values[0]
    for v in values[1:]:
        if v == prev + 1: prev = v
        else:
            ranges.append((start, prev)); start = prev = v
    ranges.append((start, prev))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a,b in ranges)

def load_csv(path: Path, strict_seq: bool=True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not path.exists(): raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing: raise ValueError(f"{path} missing columns {missing}; found {list(df.columns)}")
    df = df[REQUIRED_COLUMNS].copy()
    for c in REQUIRED_COLUMNS: df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = df[df[REQUIRED_COLUMNS].isna().any(axis=1)]
    if len(bad): raise ValueError(f"{path} has {len(bad)} bad/non-numeric rows.")
    df["seq"] = df["seq"].astype(int); df["t_ms"] = df["t_ms"].astype(int)
    seq_min, seq_max = int(df.seq.min()), int(df.seq.max())
    expected = set(range(seq_min, seq_max + 1)); seen = set(map(int, df.seq.to_numpy()))
    missing_seq = sorted(expected - seen)
    dup = int(df.seq.duplicated().sum())
    oo = int((df.seq.diff().fillna(1) < 0).sum())
    val = {"path":str(path),"rows":int(len(df)),"seq_min":seq_min,"seq_max":seq_max,
           "missing_seq_count":len(missing_seq),"missing_seq_ranges":compact_ranges(missing_seq),
           "duplicate_seq_count":dup,"out_of_order_count":oo,
           "seq_pass":len(missing_seq)==0 and dup==0 and oo==0}
    if strict_seq and not val["seq_pass"]:
        raise ValueError(f"Sequence check failed for {path}: missing {val['missing_seq_ranges']}, dup={dup}, out_of_order={oo}")
    return df, val

def trim(df: pd.DataFrame, drop_s: float) -> pd.DataFrame:
    if drop_s <= 0: return df.copy()
    keep_from = int(df.t_ms.iloc[0]) + int(round(drop_s*1000))
    out = df[df.t_ms >= keep_from].copy()
    if len(out) < 100: raise ValueError(f"Too few rows after trim: {len(out)}")
    return out

def stats(df: pd.DataFrame) -> Dict[str, Any]:
    a = df[ACCEL_COLS].to_numpy(float)
    n = np.linalg.norm(a, axis=1)
    m = df[ACCEL_COLS].mean(); s = df[ACCEL_COLS].std(ddof=1)
    return {"samples":int(len(df)),
            "mean_g":{"ax":float(m.ax_g),"ay":float(m.ay_g),"az":float(m.az_g)},
            "std_g":{"ax":float(s.ax_g),"ay":float(s.ay_g),"az":float(s.az_g)},
            "norm_g":{"mean":float(n.mean()),"std":float(n.std(ddof=1)),"min":float(n.min()),"max":float(n.max())}}

def mode_check(args):
    df0, val = load_csv(args.input, strict_seq=not args.allow_bad_seq)
    df = trim(df0, args.drop_first_s)
    st = stats(df)
    flags=[]
    if abs(st["norm_g"]["mean"]-1.0)>0.05: flags.append("Mean acceleration norm is more than 0.05 g away from 1 g.")
    if st["norm_g"]["std"]>0.02: flags.append("Acceleration norm std is high; container may have moved/vibrated.")
    out = {"stage":"stage1_container_accelerometer_check","created_utc":datetime.utcnow().replace(microsecond=0).isoformat()+"Z",
           "source_csv":str(args.input),"processing":{"drop_first_seconds":args.drop_first_s,"samples_used":len(df)},
           "recording_validation":val,"accel_stats":st,"quality":{"pass":not flags,"flags":flags},
           "interpretation":"One-orientation sanity check only. Use six-position mode for full offset/scale calibration."}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved Stage 1 accel check: {args.out}")
    print(f"Mean accel: ax={st['mean_g']['ax']:.6f}, ay={st['mean_g']['ay']:.6f}, az={st['mean_g']['az']:.6f} g")
    print(f"Norm: mean={st['norm_g']['mean']:.6f} g, std={st['norm_g']['std']:.6f} g")
    print("Quality:", "PASS" if not flags else "WARNING")
    for f in flags: print(" -", f)

def mean_for(path, drop_s, allow_bad_seq):
    df0, val = load_csv(path, strict_seq=not allow_bad_seq)
    df = trim(df0, drop_s)
    st = stats(df)
    return np.array([st["mean_g"]["ax"], st["mean_g"]["ay"], st["mean_g"]["az"]], float), val, st

def mode_six(args):
    files={"x_plus":args.x_plus,"x_minus":args.x_minus,"y_plus":args.y_plus,"y_minus":args.y_minus,"z_plus":args.z_plus,"z_minus":args.z_minus}
    means={}; vals={}; rawstats={}
    for name,path in files.items():
        means[name], vals[name], rawstats[name] = mean_for(path, args.drop_first_s, args.allow_bad_seq)
    xp,xm = means["x_plus"][0], means["x_minus"][0]
    yp,ym = means["y_plus"][1], means["y_minus"][1]
    zp,zm = means["z_plus"][2], means["z_minus"][2]
    offset = np.array([(xp+xm)/2, (yp+ym)/2, (zp+zm)/2], float)
    denom = np.array([xp-xm, yp-ym, zp-zm], float)
    if np.any(np.abs(denom)<0.1): raise ValueError("One +/− axis difference is too small. Check orientation labels.")
    scale = 2.0/denom
    expected={"x_plus":np.array([1,0,0.]),"x_minus":np.array([-1,0,0.]),"y_plus":np.array([0,1,0.]),"y_minus":np.array([0,-1,0.]),"z_plus":np.array([0,0,1.]),"z_minus":np.array([0,0,-1.])}
    residuals={}; warnings=[]
    for name,vec in means.items():
        corr=(vec-offset)*scale; err=corr-expected[name]
        residuals[name]={"raw_mean_g":{"ax":float(vec[0]),"ay":float(vec[1]),"az":float(vec[2])},
                         "corrected_mean_g":{"ax":float(corr[0]),"ay":float(corr[1]),"az":float(corr[2])},
                         "expected_g":{"ax":float(expected[name][0]),"ay":float(expected[name][1]),"az":float(expected[name][2])},
                         "error_g":{"ax":float(err[0]),"ay":float(err[1]),"az":float(err[2]),"norm":float(np.linalg.norm(err))}}
    max_err=max(v["error_g"]["norm"] for v in residuals.values())
    if np.any(scale<0): warnings.append("Negative scale factor detected. A + and - file may be swapped, or axis sign convention differs.")
    if max_err>0.08: warnings.append(f"Large post-calibration residual: {max_err:.4f} g.")
    result={"stage":"stage1_container_six_position_accelerometer_calibration","created_utc":datetime.utcnow().replace(microsecond=0).isoformat()+"Z",
            "usage_note":"Apply corrected_accel = (raw_accel_g - accel_offset_g) * accel_scale_factor.",
            "processing":{"drop_first_seconds":args.drop_first_s},"input_files":{k:str(v) for k,v in files.items()},
            "recording_validation":vals,"raw_orientation_stats":rawstats,
            "accel_offset_g":{"ax":float(offset[0]),"ay":float(offset[1]),"az":float(offset[2])},
            "accel_scale_factor":{"ax":float(scale[0]),"ay":float(scale[1]),"az":float(scale[2])},
            "six_position_residuals":residuals,"quality":{"pass":not warnings,"warnings":warnings,"max_error_norm_g":float(max_err)}}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved Stage 1 six-position accel calibration: {args.out}")
    print(f"Offset[g]: ax={offset[0]:.6f}, ay={offset[1]:.6f}, az={offset[2]:.6f}")
    print(f"Scale:     ax={scale[0]:.6f}, ay={scale[1]:.6f}, az={scale[2]:.6f}")
    print(f"Max residual={max_err:.6f} g | Quality:", "PASS" if not warnings else "WARNING")
    for w in warnings: print(" -", w)

def mode_apply(args):
    df, val = load_csv(args.input, strict_seq=not args.allow_bad_seq)
    cal = json.loads(args.calib.read_text(encoding="utf-8"))
    offset=np.array([cal["accel_offset_g"]["ax"],cal["accel_offset_g"]["ay"],cal["accel_offset_g"]["az"]], float)
    scale=np.array([cal["accel_scale_factor"]["ax"],cal["accel_scale_factor"]["ay"],cal["accel_scale_factor"]["az"]], float)
    raw=df[ACCEL_COLS].to_numpy(float); corr=(raw-offset.reshape(1,3))*scale.reshape(1,3)
    df["ax_raw_g"],df["ay_raw_g"],df["az_raw_g"]=df["ax_g"],df["ay_g"],df["az_g"]
    df["ax_g"],df["ay_g"],df["az_g"]=corr[:,0],corr[:,1],corr[:,2]
    args.out.parent.mkdir(parents=True, exist_ok=True); df.to_csv(args.out,index=False)
    print(f"\nApplied Stage 1 accel calibration: {args.out}")
    print("Raw accel columns preserved: ax_raw_g, ay_raw_g, az_raw_g")

def main():
    p=argparse.ArgumentParser(description="Stage 1 container accelerometer check/six-position calibration for BMI160 ACK/NACK CSV files.")
    sub=p.add_subparsers(dest="mode", required=True)
    c=sub.add_parser("check"); c.add_argument("--input",required=True,type=Path); c.add_argument("--out",default=Path("stage1_container_accel_check.json"),type=Path); c.add_argument("--drop-first-s",default=1.0,type=float); c.add_argument("--allow-bad-seq",action="store_true"); c.set_defaults(func=mode_check)
    s=sub.add_parser("six"); 
    for arg in ["x-plus","x-minus","y-plus","y-minus","z-plus","z-minus"]: s.add_argument("--"+arg, required=True, type=Path)
    s.add_argument("--out",default=Path("stage1_container_accel_calibration.json"),type=Path); s.add_argument("--drop-first-s",default=1.0,type=float); s.add_argument("--allow-bad-seq",action="store_true"); s.set_defaults(func=mode_six)
    a=sub.add_parser("apply"); a.add_argument("--input",required=True,type=Path); a.add_argument("--calib",required=True,type=Path); a.add_argument("--out",required=True,type=Path); a.add_argument("--allow-bad-seq",action="store_true"); a.set_defaults(func=mode_apply)
    args=p.parse_args(); args.func(args)

if __name__=="__main__": main()

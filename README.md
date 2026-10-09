# Foot-Mounted IMU Estimation of Minimum Toe Clearance

Final Year Project, Robotics and Mechatronic Engineering, Monash University (2026).
Supervisor: Dr Alpha Agape Gopalai.

Minimum toe clearance (MTC) is the smallest gap between the toe and the ground during the swing phase of walking, and is closely linked to trip and fall risk. It is normally measured with optical motion capture (MoCap), which only works in a lab. This project estimates MTC from a single foot-mounted IMU instead. It also includes a low-cost wearable IMU (BMI160 + Seeed XIAO nRF52840 Sense) that I designed, built and programmed.

📄 **Paper:** [paper/FYP_Final_Paper.pdf](paper/FYP_Final_Paper.pdf)

## Results

The algorithm was validated against synchronised MoCap from the [SensorPositionComparison2019](https://mad-lab-fau.github.io/gaitmap-datasets/auto_examples/sensor_position_comparison_2019.html) dataset, using participant-held-out gait-event detection on 12 participants and all 7 walking trials.

| Matched strides | Bias (IMU − MoCap) | MAE | RMSE | Pearson r |
|---:|---:|---:|---:|---:|
| 7142 | −3.0 mm | 8.64 mm | 11.40 mm | 0.674 |

- **Gait events:** the toe-off and initial-contact detector matched 81.3% of reference events, with a mean absolute timing error of 12.5 ms (toe-off) and 9.9 ms (initial contact).
- **Best walking condition:** normal-speed walking over 20 m gave the best agreement, with an MAE of 7.4 mm and an RMSE of 9.7 mm.
- **Custom hardware:** the BMI160/XIAO prototype captured repeatable gait-cycle gyroscope peaks that change with walking speed, compared against SPC2019 right-instep recordings. This shows the signals are plausible; the prototype's own MTC accuracy has not been validated yet.

## Pipeline

1. **Gait-event detection (Stage A EventNet):** a compact 1D dilated CNN (kernel 7, dilations 1/2/4) that labels stance and swing from 8 IMU channels, followed by hysteresis to recover toe-off (TC) and initial-contact (IC) events.
2. **Cadence-adaptive filtering:** zero-phase Butterworth filters with cut-offs scaled from the step frequency: `acc = clip(8·f_step, 10, 35) Hz`, `gyr = clip(6·f_step, 8, 30) Hz`.
3. **Orientation and gravity removal:** a Madgwick filter, stance-attitude correction, and ZUPT (zero-velocity) / minimum-velocity windows.
4. **Stride-local vertical integration:** double integration of vertical acceleration from TC to IC, with endpoint closure to suppress drift.
5. **Virtual toe:** the IMU-to-toe offset is calibrated against MoCap, and MTC is taken as the minimum of toe height above a ground level estimated from stance.
6. **Validation:** monotone dynamic-programming stride matching against MoCap, reporting bias, MAE, RMSE and Pearson r.

## Repository layout

| Path | Contents |
|---|---|
| [`algorithm/mtc_pipeline.py`](algorithm/mtc_pipeline.py) | Core signal processing: phases 1.1–1.9 (loading, filtering, events, orientation, integration, MTC, validation) |
| [`algorithm/pipeline_core.py`](algorithm/pipeline_core.py) | Wrapper that runs the phases in order and caches intermediate results |
| [`algorithm/calibrate_toe_offset.py`](algorithm/calibrate_toe_offset.py) | Coarse-to-fine IMU-to-toe offset calibration |
| [`algorithm/evaluate.py`](algorithm/evaluate.py) | Runs the pipeline with fixed (frozen) parameters and writes a summary CSV |
| [`algorithm/train_eventnet_stageA.py`](algorithm/train_eventnet_stageA.py) | EventNet training; trained weights in [`param_stageA.pt`](algorithm/param_stageA.pt) |
| [`algorithm/metrics_from_imu_only_summary.py`](algorithm/metrics_from_imu_only_summary.py) | Aggregates per-trial summaries into per-participant metrics |
| [`algorithm/experiments/`](algorithm/experiments/) | Upper-bound baseline that uses MoCap gait events |
| [`algorithm/tools/`](algorithm/tools/) | Early helper scripts: batch runner over participants, comparison CSV export, error statistics |
| [`hardware/enclosure_cad/`](hardware/enclosure_cad/) | SolidWorks enclosure design, V1 to V7 (`.SLDPRT` parts, `.STL` exports, V2 assembly); V7 is the final casing. GitHub previews the `.STL` files in 3D |
| [`hardware/firmware/`](hardware/firmware/) | XIAO nRF52840 Arduino sketches; `GyroAndAccel_Recording_V2` is the final recording firmware |
| [`hardware/host/`](hardware/host/) | Bluetooth Low Energy (BLE) logger using a three-way ACK/NACK handshake, plus stage 0–2 IMU calibration and the recordings it was fitted from (`calibration/data/`) |
| [`waveform_comparison/`](waveform_comparison/) | Comparison of BMI160 and SPC2019 waveforms; `plots/` is v2 (gaitmap frame), `plots_v1/` is the first version |
| [`data/bmi160_recordings/`](data/bmi160_recordings/) | Raw and calibrated walking recordings from the custom device |
| [`results/`](results/) | Per-trial summaries, toe-offset calibrations and stride-level error exports |

The SPC2019 dataset is not included. Download it from the link above and pass its folder as `--data_folder`.

## Running

```bash
pip install -r requirements.txt
cd algorithm

# Train the gait-event detector (hold out one participant for validation)
python train_eventnet_stageA.py --data_folder path/to/sensorpositoncomparison-v1.0.0-beta --val_participant 4d91

# Calibrate the toe offset for one participant and side
python calibrate_toe_offset.py --data_folder path/to/... --participant 4d91 --side left \
    --event_source ml --event_model_pt param_stageA.pt

# Evaluate with the calibrated offsets
python evaluate.py --data_folder path/to/... --toe_offset_csv ../results/toe_offset_calibration_xz.csv \
    --event_source ml --event_model_pt param_stageA.pt
```

## Status of the code

This repository contains the most recent versions of the scripts that I still have: the algorithm as of March 2026 and the hardware and waveform-comparison code as of May 2026. The paper's final reported run included further changes that are not in these files:

- strict-nested training with one EventNet checkpoint per held-out participant
- a transition-aware loss with TC/IC edge outputs
- a least-squares seed for the two-stage toe-offset calibration
- the 12-participant waveform comparison

As a result, rerunning this code will not reproduce the paper's tables exactly.

## Commit history

This project was developed without git. The commit history was rebuilt afterwards from dated file snapshots: each earlier version of a script became a commit, dated with that file's last-modified time. Running `git log --follow algorithm/mtc_pipeline.py` shows the pipeline's development from the first IMU-only version (January 2026) to the latest one. Intermediate commits are snapshots and may not run on their own.

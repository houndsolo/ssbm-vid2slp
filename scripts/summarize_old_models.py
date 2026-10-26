"""
summarize_old_models.py - print the headline held-out numbers of the old project's two models.

READ-ONLY. Reads copies of the old result JSON files saved in reference/old-results/
(the originals in /home/prompter/yt2slp/.docker-state were deleted on 2026-10-26).

  root-tracking-context-v1   : video frames -> pixel position of each character ("root")
  root-controller-expanded-v1: video frames -> controller inputs (buttons, stick bins)

Both report a "validation" split of whole clips/replays not used for training.
Neither ever evaluated its held-back test split ("final_test_matches_opened": 0).

Run:
    .venv/bin/python scripts/summarize_old_models.py
"""

import json

RUNS = "reference/old-results"


def tracking():
    r = json.load(open(f"{RUNS}/root-tracking-experiment-result.json"))
    p = json.load(open(f"{RUNS}/root-tracking-paired-result.json"))
    print("=== ROOT TRACKING (predicts each character's on-screen pixel position)")
    print("    video is 642x528 px; 'PCK16' = share of predictions within 16 px of the truth")
    rows = [
        ("single-frame source model", r["source_metrics"]),
        ("5 copies of centre frame", p["center"]["metrics"]),
        ("5 real frames (final)", r["metrics"]),
    ]
    for name, m in rows:
        for split in ("train", "validation"):
            s = m[split]
            print(f"  {name:28s} {split:10s} n={s['compared']:6d}  mean {s['mean_pixels']:6.2f} px  "
                  f"median {s['median_pixels']:5.2f} px  PCK8 {s['pck8']:.3f}  PCK16 {s['pck16']:.3f}  "
                  f"PCK32 {s['pck32']:.3f}  (guess-image-centre baseline {s['image_center_mean_pixels']:.0f} px)")
    print(f"  development_criteria_met={r['development_criteria_met']}  "
          f"controller_improvement_established={r['controller_improvement_established']}  "
          f"reconstructed_gameplay_validated={r['reconstructed_gameplay_validated']}")


def controller():
    r = json.load(open(f"{RUNS}/root-controller-experiment-result.json"))
    print("\n=== ROOT CONTROLLER (predicts controller inputs from video), validation split")
    arms = {"reference (earlier model)": r["reference"].get("metrics", r["reference"])}
    arms.update({f"arm '{k}'": v for k, v in r["metrics"].items()})
    for name, m in arms.items():
        if "button_metrics" not in m:
            continue
        b, s = m["button_metrics"], m["state_metrics"]
        print(f"  {name}")
        print(f"    buttons held  F1 {b['hold']['f1']:.3f}   press F1 {b['press']['f1']:.3f}   "
              f"release F1 {b['release']['f1']:.3f}   (tp/fp/fn for press: "
              f"{b['press']['tp']}/{b['press']['fp']}/{b['press']['fn']})")
        for axis in ("axis_stick_x", "axis_stick_y", "axis_cstick_x", "axis_cstick_y", "axis_trigger"):
            a = s[axis]
            print(f"    {axis:14s} accuracy {a['accuracy']:.3f}   always-neutral baseline "
                  f"{a['neutral_baseline_accuracy']:.3f}   gain {a['accuracy'] - a['neutral_baseline_accuracy']:+.3f}")
        print(f"    action-state acc {s['action']['accuracy']:.3f}   airborne acc {s['airborne']['accuracy']:.3f}   "
              f"windows {m['windows']}")
    print(f"  criteria_met={r['criteria_met']}  development_controller_gain={r['development_controller_gain']}  "
          f"reconstructed_gameplay_validated={r['reconstructed_gameplay_validated']}  "
          f"final_test_matches_opened={r['final_test_matches_opened']}")
    print("  checks:", json.dumps(r["checks"])[:800])


if __name__ == "__main__":
    tracking()
    controller()

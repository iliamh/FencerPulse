"""Generate synthetic demo dataset for fencing weapon recommendation.

This dataset is ONLY for local testing.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.engines.recommender import NUM_COLS, CAT_COLS

rng = np.random.default_rng(24)

def sample_row():
    age = int(rng.integers(13, 23))
    height = float(rng.normal(172, 9))
    weight = float(rng.normal(68, 12))
    reach = float(height * rng.uniform(0.95, 1.06))
    sprint = float(np.clip(rng.normal(3.35, 0.38), 2.55, 4.80))  # 20m
    reaction = float(np.clip(rng.normal(265, 50), 170, 450))
    beep = float(np.clip(rng.normal(8.7, 2.1), 4, 14))
    jump = float(np.clip(rng.normal(46, 11), 18, 85))
    train_h = float(np.clip(rng.normal(3.6, 2.2), 0, 12))

    dominant_hand = rng.choice(["راست", "چپ"], p=[0.86, 0.14])
    injury = rng.choice(["ندارد", "زانو", "مچ پا", "شانه", "مچ دست"], p=[0.70, 0.10, 0.08, 0.06, 0.06])
    goal = rng.choice(["تفریح", "مسابقه", "بورسیه"], p=[0.50, 0.40, 0.10])
    experience = rng.choice(["مبتدی", "متوسط", "حرفه‌ای"], p=[0.60, 0.32, 0.08])

    return {
        "age": age,
        "height_cm": round(height, 1),
        "weight_kg": round(weight, 1),
        "reach_cm": round(reach, 1),
        "sprint_20m_s": round(sprint, 2),
        "reaction_ms": round(reaction, 0),
        "beep_level": round(beep, 1),
        "jump_cm": round(jump, 0),
        "weekly_training_h": round(train_h, 1),
        "dominant_hand": dominant_hand,
        "injury": injury,
        "goal": goal,
        "experience": experience,
    }

def label(row):
    sabre = (4.8 - row["sprint_20m_s"]) * 2.0 + (450 - row["reaction_ms"]) / 90 + row["jump_cm"] / 35
    epee = row["reach_cm"] / 175 + (450 - row["reaction_ms"]) / 120 + row["beep_level"] / 9
    foil = row["weekly_training_h"] / 4 + (1.0 if row["experience"] != "مبتدی" else 0.0) + (4.4 - row["sprint_20m_s"])
    scores = np.array([foil, epee, sabre])
    return int(np.argmax(scores))

def main(n=2400):
    rows = [sample_row() for _ in range(n)]
    y = [label(r) for r in rows]
    df = pd.DataFrame(rows)
    df["label"] = y

    out_dir = os.path.join("data", "generated")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fencing_demo.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    main()

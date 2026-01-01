"""Train model and save to models/weapon_model.joblib"""
import os, sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.engines.recommender import WeaponRecommender, NUM_COLS, CAT_COLS

def main():
    data_path = os.path.join("data", "generated", "fencing_demo.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError("Run first: python scripts/make_demo_data.py")

    df = pd.read_csv(data_path)
    X = df[NUM_COLS + CAT_COLS]
    y = df["label"]
    model = WeaponRecommender().fit(X, y)

    os.makedirs("models", exist_ok=True)
    out = os.path.join("models", "weapon_model.joblib")
    model.save(out)
    print(f"Saved model to: {out}")

if __name__ == "__main__":
    main()

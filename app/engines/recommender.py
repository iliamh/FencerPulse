"""FencerPulse — Fencing weapon recommender (Foil/Epee/Sabre)

Clean, lightweight, and explainable:
- Model: L1-regularized Logistic Regression (sparse, fast on CPU).
- Explainability: show top positive contributors to the chosen class (feature weights).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

WEAPONS = ["فلوره (Foil)", "اپه (Epee)", "سابر (Sabre)"]

NUM_COLS = [
    "age",
    "height_cm",
    "weight_kg",
    "reach_cm",
    "sprint_20m_s",
    "reaction_ms",
    "beep_level",
    "jump_cm",
    "weekly_training_h",
]
CAT_COLS = [
    "dominant_hand",
    "injury",
    "goal",
    "experience",
]

@dataclass
class RecResult:
    top3: List[Tuple[str, float]]
    primary: str
    confidence: float
    explanation_items: List[Tuple[str, float]]  # (feature, contribution)

class WeaponRecommender:
    def __init__(self) -> None:
        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=[("scaler", StandardScaler())]), NUM_COLS),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ],
            remainder="drop",
            sparse_threshold=0.3,
        )

        clf = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=1.0,
            max_iter=3000,
            n_jobs=1,
            random_state=7,
            multi_class="ovr",
        )

        self.pipeline = Pipeline(steps=[("pre", pre), ("clf", clf)])
        self._feature_names: List[str] | None = None

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "WeaponRecommender":
        self.pipeline.fit(df[NUM_COLS + CAT_COLS], y)
        self._feature_names = self._get_feature_names()
        return self

    def predict(self, x: Dict) -> RecResult:
        df = pd.DataFrame([x])
        proba = self.pipeline.predict_proba(df[NUM_COLS + CAT_COLS])[0]
        order = np.argsort(proba)[::-1]
        top3 = [(WEAPONS[i], float(proba[i])) for i in order[:3]]
        best = int(order[0])

        items = self._explain(df, class_index=best, topk=6)
        return RecResult(
            top3=top3,
            primary=WEAPONS[best],
            confidence=float(proba[best]),
            explanation_items=items,
        )

    def _get_feature_names(self) -> List[str]:
        pre: ColumnTransformer = self.pipeline.named_steps["pre"]
        num_names = NUM_COLS
        cat: OneHotEncoder = pre.named_transformers_["cat"]
        cat_names = list(cat.get_feature_names_out(CAT_COLS))
        return num_names + cat_names

    def _explain(self, df: pd.DataFrame, class_index: int, topk: int = 6) -> List[Tuple[str, float]]:
        pre: ColumnTransformer = self.pipeline.named_steps["pre"]
        clf: LogisticRegression = self.pipeline.named_steps["clf"]

        X = pre.transform(df[NUM_COLS + CAT_COLS])
        Xd = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

        coef = clf.coef_[class_index]
        contrib = Xd[0] * coef

        names = self._feature_names or [f"f{i}" for i in range(len(contrib))]
        pairs = list(zip(names, contrib.tolist()))
        pairs.sort(key=lambda t: abs(t[1]), reverse=True)

        mapping = {
            "age": "سن",
            "height_cm": "قد",
            "weight_kg": "وزن",
            "reach_cm": "ریچ (طول دست)",
            "sprint_20m_s": "زمان ۲۰ متر",
            "reaction_ms": "واکنش (ms)",
            "beep_level": "بیپ تست",
            "jump_cm": "پرش عمودی",
            "weekly_training_h": "ساعت تمرین هفتگی",
            "dominant_hand": "دست غالب",
            "injury": "آسیب‌دیدگی",
            "goal": "هدف",
            "experience": "تجربه",
        }

        pretty: List[Tuple[str, float]] = []
        for name, val in pairs[:topk]:
            for k, v in mapping.items():
                name = name.replace(k, v)
            name = name.replace("_", " = ")
            pretty.append((name, float(val)))
        return pretty

    def save(self, path: str) -> None:
        dump({"pipeline": self.pipeline, "feature_names": self._feature_names}, path)

    @staticmethod
    def load(path: str) -> "WeaponRecommender":
        obj = load(path)
        inst = WeaponRecommender()
        inst.pipeline = obj["pipeline"]
        inst._feature_names = obj.get("feature_names")
        return inst

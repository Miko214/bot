#!/usr/bin/env python3

import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names*",
    category=UserWarning
)

import glob
import os
import json
import pandas as pd
import joblib
from datetime import datetime


def fetch_current_features() -> dict:
    # TODO: вернуть здесь реальные фичи в виде {feature_name: value}
    raise NotImplementedError


def load_latest_model():
    """
    Ищет в папке 'models' файлы trade_model_*.pkl
    и возвращает модель + список фич из самого свежего.
    """
    model_dir = "models"
    pattern = "trade_model_*.pkl"
    files = glob.glob(os.path.join(model_dir, pattern))
    if not files:
        raise FileNotFoundError(f"В папке '{model_dir}' нет ни одного файла '{pattern}'")
    latest = max(files, key=os.path.getmtime)
    print(f"Loading latest model: {latest}")
    mdl = joblib.load(latest)
    return mdl["model"], mdl["features"]


def main():
    # 1) Читаем порог из config.json
    cfg = json.load(open("config.json", "r"))
    thr = cfg.get("filter_threshold", 0.6)

    # 2) Загружаем модель и фичи из папки models/
    model, features = load_latest_model()

    # 3) Собираем текущий вектор фич
    candle = fetch_current_features()
    missing = [f for f in features if f not in candle]
    if missing:
        raise KeyError(f"Отсутствуют фичи: {missing}")

    # 4) Строим DataFrame в том же порядке колонок
    X_new = pd.DataFrame([{f: candle[f] for f in features}], columns=features)

    # 5) Предсказываем вероятность и выводим сигнал
    prob = model.predict_proba(X_new)[0, 1]
    now = datetime.utcnow().isoformat()
    print(f"{now} ▶ prob = {prob:.4f} (thr={thr})")

    if prob > thr:
        print("✅ SIGNAL")
    else:
        print("❌ NO SIGNAL")


if __name__ == "__main__":
    main()

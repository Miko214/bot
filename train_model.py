import os
import warnings
import logging
from datetime import datetime, timezone

import pandas as pd
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

# ИСПРАВЛЕНИЕ: Заменяем train_test_split на TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit # НОВЫЙ ИМПОРТ
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score # Добавляем больше метрик
from imblearn.over_sampling import SMOTE # Импорт SMOTE

# Подавляем предупреждения и настраиваем логирование
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("lightgbm").setLevel(logging.ERROR)

def main():
    # 1) Загружаем данные
    df = pd.read_csv("training_dataset.csv")
    
    # Проверка и преобразование 'entry_time' (теперь она должна быть в файле)
    if 'entry_time' not in df.columns:
        raise ValueError("Колонка 'entry_time' отсутствует в training_dataset.csv. Она необходима для TimeSeriesSplit.")
    
    df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True) # Добавляем utc=True для консистентности
    df.sort_values(by=['symbol', 'entry_time'], inplace=True) # Важно сортировать по времени для TimeSeriesSplit

    print(f"Итоговые данные для обучения: {len(df)} строк.")
    print("Распределение меток в исходных данных: ")
    print(df['label'].value_counts())

    # Исключаем нечисловые колонки и целевую переменную 'label' из признаков X
    # 'entry_time' используется для TimeSeriesSplit, но не как признак для модели
    # 'symbol' и 'side' - категориальные и не могут быть напрямую поданы в VarianceThreshold
    X_cols_to_drop = ['entry_time', 'symbol', 'side', 'label']
    X = df.drop(columns=[col for col in X_cols_to_drop if col in df.columns], errors='ignore')
    y = df['label']

    # --- Отбор признаков (VarianceThreshold) ---
    # Удаляем признаки с нулевой дисперсией (константные признаки)
    selector = VarianceThreshold()
    X_selected = selector.fit_transform(X) # <-- Ошибка возникала здесь
    
    # Получаем имена оставшихся признаков
    selected_feature_names = X.columns[selector.get_support()].tolist()
    print(f"Выбрано {len(selected_feature_names)} признаков после VarianceThreshold.")
    print(f"Выбранные признаки: {selected_feature_names}")

    # Преобразуем X_selected обратно в DataFrame с правильными именами колонок
    X = pd.DataFrame(X_selected, columns=selected_feature_names, index=X.index)

    # --- Подготовка для обучения: TimeSeriesSplit ---
    tscv = TimeSeriesSplit(n_splits=5) # 5 фолдов для временных рядов
    
    optimal_thresholds = [] # Для сбора оптимальных порогов из каждого фолда
    roc_aucs = []           # Для сбора ROC AUC из каждого фолда

    # Добавлено для графиков ROC
    plt.figure(figsize=(10, 8))
    lw = 2

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\n--- ФОЛД {fold + 1} ---")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        print(f"Размер обучающего набора: {len(X_train)} строк, размер тестового набора: {len(X_test)} строк.")
        print(f"Распределение меток в обучающем наборе:\n{y_train.value_counts()}")
        print(f"Распределение меток в тестовом наборе:\n{y_test.value_counts()}")

        # Проверка на наличие хотя бы одного класса в обучающем наборе
        if len(y_train.unique()) < 2:
            print(f"Пропускаем фолд {fold + 1}: В обучающем наборе недостаточно классов для SMOTE или обучения.")
            continue

        # --- Обработка дисбаланса классов с помощью SMOTE ---
        # Применяем SMOTE только к обучающему набору
        try:
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            print(f"Размер обучающего набора после SMOTE: {len(X_train_res)} строк.")
            print(f"Распределение меток после SMOTE:\n{y_train_res.value_counts()}")
        except ValueError as e:
            print(f"SMOTE не может быть применен: {e}. Продолжаем без SMOTE.")
            X_train_res, y_train_res = X_train, y_train # Используем исходные данные

        # --- Обучение модели LightGBM ---
        lgb_clf = lgb.LGBMClassifier(random_state=42, n_estimators=1000, learning_rate=0.05, num_leaves=31)
        lgb_clf.fit(X_train_res, y_train_res, 
                    eval_set=[(X_test, y_test)], 
                    eval_metric='auc', 
                    callbacks=[lgb.early_stopping(100, verbose=False)])

        # --- Оценка модели ---
        y_pred_proba = lgb_clf.predict_proba(X_test)[:, 1]
        y_pred = lgb_clf.predict(X_test) # Для report и других метрик

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        roc_aucs.append(roc_auc)
        print(f"ROC AUC для фолда {fold + 1}: {roc_auc:.4f}")

        # Вывод отчета по классификации
        print(f"Отчет по классификации для фолда {fold + 1}:\n{classification_report(y_test, y_pred)}")

        # Поиск оптимального порога
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        J = tpr - fpr # Индекс Юдена
        ix = np.argmax(J)
        best_thr = thresholds[ix]
        optimal_thresholds.append(best_thr)
        print(f"Оптимальный порог для фолда {fold + 1} (максимум индекса Юдена): {best_thr:.4f}")
        
        # Добавляем кривую ROC для текущего фолда
        plt.plot(fpr, tpr, lw=lw, label=f'ROC fold {fold + 1} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Сохраняем график
    roc_plot_filename = "roc_curve_folds.png"
    plt.savefig(roc_plot_filename)
    print(f"\nГрафик ROC-кривой для всех фолдов сохранен как {roc_plot_filename}")
    plt.close() # Закрываем фигуру, чтобы предотвратить отображение в консоли

    if roc_aucs:
        print(f"\nСредний ROC AUC по всем фолдам: {np.mean(roc_aucs):.4f}")
    
    if optimal_thresholds:
        avg_optimal_thr = np.mean(optimal_thresholds)
        print(f"Средний оптимальный порог по всем фолдам: {avg_optimal_thr:.4f}")

    # --- Финальное обучение на всех данных ---
    print("\n--- Финальное обучение модели на всех доступных данных ---")
    
    # Снова применяем SMOTE ко всему датасету, если он был эффективен ранее
    try:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        print(f"Размер полного обучающего набора после SMOTE: {len(X_res)} строк.")
        print(f"Распределение меток после SMOTE:\n{y_res.value_counts()}")
    except ValueError as e:
        print(f"SMOTE не может быть применен ко всему набору: {e}. Продолжаем без SMOTE.")
        X_res, y_res = X, y # Используем исходные данные

    final_model = lgb.LGBMClassifier(random_state=42, n_estimators=1000, learning_rate=0.05, num_leaves=31)
    final_model.fit(X_res, y_res)

    # Оценка финальной модели на всем наборе (для проверки)
    final_y_pred_proba = final_model.predict_proba(X)[:, 1]
    final_y_pred = final_model.predict(X)

    best_thr_final = 0.5 # По умолчанию
    if len(np.unique(y)) > 1: # Убедимся, что есть оба класса для ROC-анализа
        fpr_final, tpr_final, thresholds_final = roc_curve(y, final_y_pred_proba)
        J_final = tpr_final - fpr_final
        ix_final = np.argmax(J_final)
        best_thr_final = thresholds_final[ix_final]
        print(f"Оптимальный порог на всем наборе данных (максимум индекса Юдена): {best_thr_final:.4f}")
        print(f"Финальный ROC AUC на полном наборе: {roc_auc_score(y, final_y_pred_proba):.4f}")
    else:
        print("Недостаточно классов для проведения ROC-анализа на полном наборе данных.")


    # --- Сохранение модели и фичей ---
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    model_filename = os.path.join(model_dir, f"trade_model_{now}.pkl")

    # Сохраняем модель и список фич
    joblib.dump({
            "model": final_model,
            "features": selected_feature_names, # Сохраняем только отобранные фичи
            "timestamp": now
        }, model_filename)
    print(f"\n✅ Модель и список фич сохранены в {model_filename}")

    # Сохраняем лучший порог в config.json
    if optimal_thresholds:
        config_file = 'config.json'
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            cfg = {} # Если файл не найден или пуст, начинаем с пустого конфига

        # Используем итоговый порог, найденный по всей совокупности (best_thr_final)
        final_filter_threshold = best_thr_final
        
        cfg['filter_threshold'] = float(f"{final_filter_threshold:.4f}") # Округляем для конфига
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=4)
        print(f"Обновлен filter_threshold в {config_file} до {cfg['filter_threshold']}.")
    else:
        print("Не удалось определить оптимальный порог для сохранения в config.json. Используется значение по умолчанию.")

if __name__ == "__main__":
    main()
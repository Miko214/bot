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
    
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df.sort_values(by=['symbol', 'entry_time'], inplace=True) # Важно сортировать по времени для TimeSeriesSplit

    # Исключаем 'entry_time' (и 'symbol', 'history_timestamp', 'join_time') из списка признаков для модели,
    # но используем их для сортировки и в TimeSeriesSplit
    feature_cols = [c for c in df.columns if c not in ["label", "symbol", "entry_time", "exit_time", "reason", "pnl", "source", "entry_price", "exit_price", "history_timestamp", "join_time"]]
    
    # Удаляем фичи, которые могут быть полностью NaN после merge или неактуальны
    # Например, если какая-то свеча в истории была неполной
    df.dropna(subset=feature_cols, inplace=True)

    X = df[feature_cols].copy()
    y = df["label"].copy()

    print(f"Итоговые данные для обучения: {len(X)} строк.")
    print(f"Распределение меток в исходных данных: \n{y.value_counts()}")

    if len(X) == 0:
        print("Нет данных для обучения после очистки NaN. Проверьте training_dataset.csv и процесс генерации признаков.")
        return

    # 2) Убираем признаки с нулевой дисперсией (если есть)
    selector = VarianceThreshold(threshold=0.0) # Удаляет признаки, которые не меняются
    X_selected = selector.fit_transform(X)
    selected_feature_names = X.columns[selector.get_support()]
    X = pd.DataFrame(X_selected, columns=selected_feature_names, index=X.index)
    print(f"Признаков после VarianceThreshold: {len(selected_feature_names)}")

    # ИСПРАВЛЕНИЕ: Внедряем TimeSeriesSplit
    # Количество разбиений для кросс-валидации. Установите по вкусу.
    # Чем больше n_splits, тем больше моделей будет обучено.
    N_SPLITS = 5 
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    models = []
    roc_aucs = []
    optimal_thresholds = []
    
    # Для усреднения метрик по всем фолдам
    all_y_test = []
    all_y_prob = []
    
    print(f"\nНачало TimeSeries Cross-Validation с {N_SPLITS} разбиениями...")

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\n--- ФОЛД {fold + 1}/{N_SPLITS} ---")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        print(f"Размер тренировочной выборки: {len(X_train)} строк")
        print(f"Размер тестовой выборки: {len(X_test)} строк")
        print(f"Распределение меток в тренировочной выборке:\n{y_train.value_counts()}")
        print(f"Распределение меток в тестовой выборке:\n{y_test.value_counts()}")

        # 3) Если одного из классов <10 в тренировочной выборке — делаем SMOTE
        counts_train = y_train.value_counts()
        if counts_train.min() < 10:
            print("⚠️ Слишком мало примеров в тренировочной выборке — применяем SMOTE")
            # Проверяем, достаточно ли соседей для SMOTE
            k_neighbors = min(counts_train.min(), 5) # min_samples = 2 * k_neighbors
            # Убедитесь, что k_neighbors >= 1, SMOTE требует хотя бы 1 соседа
            if k_neighbors < 1:
                print("Недостаточно данных для SMOTE. Пропускаем SMOTE для этого фолда.")
                X_res, y_res = X_train, y_train
            else:
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_res, y_res = smote.fit_resample(X_train, y_train)
            print(f"Распределение меток после SMOTE:\n{y_res.value_counts()}")
        else:
            X_res, y_res = X_train, y_train
        
        # 4) Обучение модели LightGBM
        # Использование scale_pos_weight для автоматического взвешивания классов
        # Это может быть более эффективным, чем только SMOTE
        # pos_weight = (количество отрицательных примеров) / (количество положительных примеров)
        scale_pos_weight_val = counts_train[0] / counts_train[1] if 1 in counts_train and counts_train[1] > 0 else 1.0

        model = lgb.LGBMClassifier(objective='binary',
                                   metric='auc',
                                   n_estimators=1000,
                                   learning_rate=0.05,
                                   num_leaves=31,
                                   max_depth=-1,
                                   min_child_samples=20,
                                   subsample=0.8,
                                   colsample_bytree=0.8,
                                   random_state=42,
                                   n_jobs=-1,
                                   # НОВОЕ: Взвешивание классов
                                   scale_pos_weight=scale_pos_weight_val
                                  )
        
        # Ранняя остановка, чтобы избежать переобучения
        # eval_set - используем тестовую выборку текущего фолда для валидации
        model.fit(X_res, y_res,
                  eval_set=[(X_test, y_test)],
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(100, verbose=False)]) # Остановиться, если AUC не улучшается 100 итераций

        # 5) Оценка модели на тестовой выборке текущего фолда
        if len(set(y_test)) > 1: # Проверяем, что в тестовой выборке есть оба класса
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred_raw = (y_prob > 0.5).astype(int) # Прогноз по дефолтному порогу 0.5
            
            # Сохраняем для финальной агрегированной оценки
            all_y_test.extend(y_test)
            all_y_prob.extend(y_prob)

            auc_fold = roc_auc_score(y_test, y_prob)
            print(f"ФОЛД {fold + 1} ROC AUC: {auc_fold:.4f}")
            
            # Подбор оптимального порога для этого фолда
            fpr, tpr, thr = roc_curve(y_test, y_prob)
            # Избегаем inf, если порог 1.0 или 0.0
            if len(thr) > 1 and thr[0] == 1.0: # Если первый порог 1.0, это может быть проблемой
                fpr, tpr, thr = fpr[1:], tpr[1:], thr[1:]
            
            youden = tpr - fpr
            if len(youden) > 0: # Убедимся, что есть данные для Youden
                best_idx = youden.argmax()
                best_thr_fold = thr[best_idx]
                print(f"ФОЛД {fold + 1} Optimal threshold by Youden: {best_thr_fold:.4f}")
                optimal_thresholds.append(best_thr_fold)

                # Дополнительные метрики по оптимальному порогу
                y_pred_optimal = (y_prob > best_thr_fold).astype(int)
                print(f"ФОЛД {fold + 1} Accuracy (optimal thr): {accuracy_score(y_test, y_pred_optimal):.4f}")
                print(f"ФОЛД {fold + 1} Precision (optimal thr): {precision_score(y_test, y_pred_optimal, zero_division=0):.4f}")
                print(f"ФОЛД {fold + 1} Recall (optimal thr): {recall_score(y_test, y_pred_optimal, zero_division=0):.4f}")
                print(f"ФОЛД {fold + 1} F1-Score (optimal thr): {f1_score(y_test, y_pred_optimal, zero_division=0):.4f}")
            else:
                print(f"ФОЛД {fold + 1} Недостаточно точек для расчета Youden's Index.")
            
            roc_aucs.append(auc_fold)
            models.append(model) # Сохраняем модель для каждого фолда

        else:
            print(f"ФОЛД {fold + 1}: Только один класс в тестовой выборке. Пропуск оценки AUC.")

    if not models:
        print("Не удалось обучить ни одной модели или провести оценку AUC. Завершение.")
        return

    # 6) Агрегированные метрики по всем фолдам
    print("\n--- АГРЕГИРОВАННЫЕ РЕЗУЛЬТАТЫ ПО ВСЕМ ФОЛДАМ ---")
    if roc_aucs:
        print(f"Средний ROC AUC по фолдам: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
    if optimal_thresholds:
        avg_optimal_thr = np.mean(optimal_thresholds)
        std_optimal_thr = np.std(optimal_thresholds)
        print(f"Средний оптимальный порог по Youden: {avg_optimal_thr:.4f} ± {std_optimal_thr:.4f}")

        # Финальный отчет классификации по агрегированным данным с средним оптимальным порогом
        final_y_pred_optimal = (np.array(all_y_prob) > avg_optimal_thr).astype(int)
        print("\nИТОГОВЫЙ ОТЧЕТ КЛАССИФИКАЦИИ (по среднему оптимальному порогу):")
        print(classification_report(all_y_test, final_y_pred_optimal, digits=4, zero_division=0))
        
        # 7) ROC-анализ и подбор порога Youden для агрегированных данных
        if len(set(all_y_test)) > 1:
            auc_final = roc_auc_score(all_y_test, all_y_prob)
            fpr_final, tpr_final, thr_final = roc_curve(all_y_test, all_y_prob)
            
            # Убеждаемся, что пороги не содержат inf и корректны
            if len(thr_final) > 1 and thr_final[0] == 1.0:
                fpr_final, tpr_final, thr_final = fpr_final[1:], tpr_final[1:], thr_final[1:]
            
            youden_final = tpr_final - fpr_final
            if len(youden_final) > 0:
                best_idx_final = youden_final.argmax()
                best_thr_final = thr_final[best_idx_final]
                print(f"Итоговый ROC AUC (все данные): {auc_final:.4f}, Итоговый оптимальный порог по Youden: {best_thr_final:.4f}")

                plt.figure(figsize=(8,6))
                sns.lineplot(x=fpr_final, y=tpr_final, label="ROC Curve")
                plt.plot([0,1], [0,1], "--", color="gray")
                plt.scatter(fpr_final[best_idx_final], tpr_final[best_idx_final], color="red", label=f"Best thr = {best_thr_final:.2f}")
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.title(f"ИТОГОВЫЙ ROC AUC: {auc_final:.4f}")
                plt.legend()
                plt.tight_layout()
                plt.show()
            else:
                print("❗ Недостаточно точек для итогового расчета Youden's Index.")
        else:
            print("❗ Только один класс в агрегированных тестовых данных — пропуск итогового ROC-анализа.")
    else:
        print("Недостаточно данных для агрегированного ROC-анализа.")


    # 8) Сохраняем ЛУЧШУЮ модель и список фич
    # Обычно сохраняют модель, которая показала наилучший AUC на кросс-валидации
    # Или можно взять последнюю обученную модель, если TimeSeriesSplit подразумевает,
    # что она обучена на самых свежих данных.
    # В данном случае, чтобы не усложнять, сохраняем последнюю модель из цикла.
    # Для более продвинутой логики можно выбрать модель с лучшим ROC AUC
    # из списка models = []
    
    # Для простоты, сохраняем модель из последнего фолда.
    # Если вы хотите сохранить "лучшую" модель, вам нужно отслеживать
    # производительность каждой модели и сохранять ту, которая лучше.
    # Но для TimeSeriesSplit последняя модель обучена на самом большом и свежем наборе данных.
    if models:
        final_model = models[-1] # Берем модель из последнего фолда
        now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        model_filename = f"models/trade_model_{now}.pkl"

        # Сохраняем модель и список фич в одном файле
        joblib.dump({
            "model": final_model,
            "features": selected_feature_names.tolist(), # ИСправлено: используем selected_feature_names
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

            # Используем средний оптимальный порог или итоговый, найденный по всей совокупности
            # Здесь я использую итоговый порог из ROC-анализа по всем данным (best_thr_final)
            # Если roc_aucs пуст, то best_thr_final не определен, поэтому используем avg_optimal_thr
            final_filter_threshold = best_thr_final if 'best_thr_final' in locals() else (avg_optimal_thr if 'avg_optimal_thr' in locals() else 0.5)
            
            cfg['filter_threshold'] = float(f"{final_filter_threshold:.4f}") # Округляем для конфига
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=4)
            print(f"Обновлен filter_threshold в {config_file} до {cfg['filter_threshold']}.")
        else:
            print("Не удалось определить оптимальный порог для сохранения в config.json.")
    else:
        print("Модель не была обучена или сохранена.")

if __name__ == "__main__":
    main()
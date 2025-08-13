#!/usr/bin/env python3
# feature_engineering.py

import pandas as pd
import os
import numpy as np

def main():
    # 1) Загрузим размеченные сделки от бота
    labels_bot = pd.read_csv('labeled_trades.csv', dtype={'symbol': str})
    # Убедимся, что entry_time и exit_time являются datetime объектами
    labels_bot['entry_time'] = pd.to_datetime(labels_bot['entry_time'], errors='coerce')
    labels_bot['exit_time'] = pd.to_datetime(labels_bot['exit_time'], errors='coerce')
    labels_bot['source'] = 'bot'

    # 2) Загрузим экспертные сделки, если файл существует
    expert_trades_file = 'expert_trades.csv'
    labels = labels_bot.copy() # labels начинается как копия labels_bot
    if os.path.exists(expert_trades_file) and os.path.getsize(expert_trades_file) > 0:
        try:
            labels_expert = pd.read_csv(expert_trades_file, dtype={'symbol': str})
            # Убедимся, что entry_time и exit_time являются datetime объектами
            labels_expert['entry_time'] = pd.to_datetime(labels_expert['entry_time'], errors='coerce')
            labels_expert['exit_time'] = pd.to_datetime(labels_expert['exit_time'], errors='coerce')
            labels_expert['source'] = 'expert'

            labels = pd.concat([labels_bot, labels_expert], ignore_index=True)
            labels.sort_values(by=['symbol', 'entry_time', 'source'], ascending=[True, True, False], inplace=True)
            labels.drop_duplicates(subset=['symbol', 'entry_time'], keep='first', inplace=True)
            print(f"Объединено {len(labels_bot)} сделок бота и {len(labels_expert)} экспертных сделок. Итого: {len(labels)} уникальных размеченных записей.")
        except pd.errors.EmptyDataError:
            print(f"Файл {expert_trades_file} существует, но пуст. Используем только метки бота.")
        except Exception as e:
            print(f"Ошибка при загрузке {expert_trades_file}: {e}. Используем только метки бота.")
    else:
        print(f"Файл {expert_trades_file} не найден или пуст. Используем только метки бота.")
    
    if labels.empty:
        print("Нет данных для разметки. Убедитесь, что 'labeled_trades.csv' или 'expert_trades.csv' содержат данные.")
        required_cols = ['symbol', 'side', 'entry_time', 'entry_price', 'exit_time', 'exit_price', 'reason', 'pnl', 'label']
        # Создаем пустой DataFrame с нужными колонками, чтобы избежать ошибок на следующих шагах
        # Добавляем все колонки, которые могут быть в training_dataset.csv, включая фичи
        pd.DataFrame(columns=['symbol', 'entry_time', 'label'] + [col for col in required_cols if col not in ['symbol', 'entry_time', 'label']]).to_csv('training_dataset.csv', index=False)
        return

    # 3) Загрузим полную историю с индикаторами
    full_history = pd.read_csv('full_history_with_indicators.csv', dtype={'symbol': str})
    # Убедимся, что timestamp является datetime объектом
    full_history['timestamp'] = pd.to_datetime(full_history['timestamp'], errors='coerce')
    print(f"Загружено {len(full_history)} строк полной истории.")

    # 4) Для удобства объединения, переименуем timestamp в истории в join_time
    labels['join_time'] = labels['entry_time']

    # --- НАЧАЛО ИСПРАВЛЕНИЙ ДЛЯ ВРЕМЕННЫХ ЗОН ---
    # Функция для нормализации временных меток в UTC
    def normalize_timestamp_to_utc(ts):
        if pd.isna(ts): # Пропускаем NaN значения
            return ts
        if ts.tzinfo is not None and ts.tzinfo.utcoffset(ts) is not None:
            # Если метка уже tz-aware, конвертируем её в UTC
            return ts.tz_convert('UTC')
        else:
            # Если метка tz-naive (без информации о часовом поясе),
            # Предполагаем, что она в UTC, и делаем её tz-aware в UTC.
            return ts.tz_localize('UTC')

    # Применяем нормализацию к 'join_time' в DataFrame 'labels'
    labels['join_time'] = labels['join_time'].apply(normalize_timestamp_to_utc)

    # Применяем нормализацию к 'timestamp' в DataFrame 'full_history'
    # Это КРАЙНЕ ВАЖНО, чтобы оба столбца для merge_asof были единообразны (все UTC, tz-aware)
    full_history['timestamp'] = full_history['timestamp'].apply(normalize_timestamp_to_utc)

    # --- КОНЕЦ ИСПРАВЛЕНИЙ ДЛЯ ВРЕМЕННЫХ ЗОН ---

    full_history.rename(columns={'timestamp': 'history_timestamp'}, inplace=True)

    # 5) Объединяем размеченные сделки с историческими данными (индикаторами)
    print("Объединение размеченных сделок с историческими данными...")
    merged_parts = []
    for sym in labels['symbol'].unique():
        labels_sym = labels[labels['symbol'] == sym].sort_values('join_time')
        history_sym = full_history[full_history['symbol'] == sym].sort_values('history_timestamp')
        
        m = pd.merge_asof(
            labels_sym,
            history_sym,
            left_on='join_time',
            right_on='history_timestamp',
            by='symbol',
            direction='backward',
            tolerance=pd.Timedelta('1m')
        )
        merged_parts.append(m)
    data = pd.concat(merged_parts, ignore_index=True)

    # 6) Удаляем строки, где не нашлось соответствия в истории
    initial_rows = len(data)
    data.dropna(subset=['history_timestamp'], inplace=True) 
    print(f"Удалено {initial_rows - len(data)} строк без соответствующей исторической свечи.")

    # --- НОВЫЕ ПРИЗНАКИ ---

    # 7) Добавляем временные признаки
    print("Добавление временных признаков (этот блок можно удалить, если признаки не нужны)...")
    # Убедимся, что 'entry_time' также tz-aware для dt.hour и т.д.
    data['entry_time'] = data['entry_time'].apply(normalize_timestamp_to_utc)

    # # УДАЛЕНЫ ВРЕМЕННЫЕ ПРИЗНАКИ ПО ЗАПРОСУ ПОЛЬЗОВАТЕЛЯ:
    # data['hour_of_day'] = data['entry_time'].dt.hour
    # data['day_of_week'] = data['entry_time'].dt.dayofweek
    # data['day_of_month'] = data['entry_time'].dt.day
    # data['month_of_year'] = data['entry_time'].dt.month
    # data['is_weekend'] = ((data['entry_time'].dt.dayofweek == 5) | (data['entry_time'].dt.dayofweek == 6)).astype(int)

    # 8) Добавляем отстающие (lagged) признаки - ЭТОТ БЛОК ПОКА НЕ МЕНЯЕМ
    print("Добавление отстающих признаков (для будущей имплементации в generate_history.py)...")

    # 9) Выбираем признаки и метку
    # ОБНОВЛЕННЫЙ СПИСОК ПРИЗНАКОВ: Удалены STOCH, CCI, VWAP и временные признаки
    feature_cols = [
        'ATR_14',
        'EMA_50', 'EMA_200',
        'RSI_14',
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'BB_upper', 'BB_middle', 'BB_lower', 'BB_width',
        'VOLUME_EMA',
        'KAMA_10_2_30',
        'CMF_20',
        'RVI_14',
        'ADX_14',
        'volume',
        'open', 'high', 'low', 'close',
        # Удалены временные признаки:
        # 'hour_of_day',
        # 'day_of_week',
        # 'day_of_month',
        # 'month_of_year',
        # 'is_weekend'
    ]

    # Фильтруем feature_cols, чтобы оставить только те, что фактически есть в DataFrame
    data_columns_set = set(data.columns)
    
    # Создаем новый список признаков, содержащий только те, которые фактически присутствуют в 'data'
    final_feature_cols = [col for col in feature_cols if col in data_columns_set]

    missing_from_expected = [col for col in feature_cols if col not in data_columns_set]
    if missing_from_expected:
        print(f"⚠️ Предупреждение: Следующие ожидаемые признаки отсутствуют в итоговом датасете и будут исключены: {missing_from_expected}.")
        # Мы уже сформировали final_feature_cols, так что просто выводим предупреждение.

    # Создаем X (признаки) и y (метки)
    X = data[final_feature_cols].copy() # Используем final_feature_cols
    y = data["label"].copy()

    # --- ОКОНЧАТЕЛЬНАЯ ОБРАБОТКА И СОХРАНЕНИЕ ---

    # Важно: перед dropna сохраняем 'entry_time' и 'symbol'
    # Используем 'data.index' для синхронизации после dropna
    entry_times_to_save = data['entry_time'].copy()
    symbols_to_save = data['symbol'].copy()

    # Объединяем X и y для совместной очистки NaN
    combined_for_dropna = pd.concat([X, y], axis=1)
    
    before_dropna_final = len(combined_for_dropna)
    combined_for_dropna = combined_for_dropna.dropna()
    print(f"После final dropna: {before_dropna_final} -> {len(combined_for_dropna)} строк.")

    # Разделяем обратно X и y после очистки
    y = combined_for_dropna.pop('label') # Извлекаем метку
    X = combined_for_dropna # Оставшееся - это признаки

    # Синхронизируем entry_times_to_save и symbols_to_save с отфильтрованными индексами
    entry_times_to_save = entry_times_to_save.loc[X.index]
    symbols_to_save = symbols_to_save.loc[X.index]

    # Объединяем X, y, entry_time и symbol в один DataFrame для сохранения
    training_df = pd.concat([X, y, entry_times_to_save, symbols_to_save], axis=1)

    # 10) Сохраняем итоговый файл
    output_training_file = 'training_dataset.csv'
    if not training_df.empty:
        training_df.to_csv(output_training_file, index=False)
        print(f"Обучающий датасет сохранен в {output_training_file}. Всего: {len(training_df)} строк.")
        print(f"Распределение меток: \n{training_df['label'].value_counts()}")
        print(f"Использованные признаки (X): {X.columns.tolist()}") 
    else:
        # Если DataFrame пуст, создаем пустой файл с заголовками
        empty_df = pd.DataFrame(columns=feature_cols + ['label', 'entry_time', 'symbol'])
        empty_df.to_csv(output_training_file, index=False)
        print(f"Не найдено данных для обучения. Создан или обновлен пустой файл {output_training_file}.")


if __name__ == "__main__":
    main()
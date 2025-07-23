#!/usr/bin/env python3
# feature_engineering.py

import pandas as pd
import os
import numpy as np

def main():
    # 1) Загрузим размеченные сделки от бота
    labels_bot = pd.read_csv('labeled_trades.csv', dtype={'symbol': str})
    # Преобразование в datetime и приведение к UTC
    labels_bot['entry_time'] = pd.to_datetime(labels_bot['entry_time'], errors='coerce', utc=True)
    labels_bot['exit_time'] = pd.to_datetime(labels_bot['exit_time'], errors='coerce', utc=True)
    labels_bot['source'] = 'bot'
    labels_bot.dropna(subset=['entry_time', 'exit_time'], inplace=True) # Удаляем строки с некорректными датами

    # 2) Загрузим экспертные сделки, если файл существует
    expert_trades_file = 'expert_trades.csv'
    labels = labels_bot.copy()
    if os.path.exists(expert_trades_file) and os.path.getsize(expert_trades_file) > 0:
        try:
            labels_expert = pd.read_csv(expert_trades_file, dtype={'symbol': str})
            # Преобразование в datetime и приведение к UTC
            labels_expert['entry_time'] = pd.to_datetime(labels_expert['entry_time'], errors='coerce', utc=True)
            labels_expert['exit_time'] = pd.to_datetime(labels_expert['exit_time'], errors='coerce', utc=True)
            labels_expert['source'] = 'expert'
            labels_expert.dropna(subset=['entry_time', 'exit_time'], inplace=True) # Удаляем строки с некорректными датами

            # Объединяем и удаляем дубликаты, предпочитая экспертные метки
            labels = pd.concat([labels_bot, labels_expert], ignore_index=True)
            labels.sort_values(by=['symbol', 'entry_time', 'source'], ascending=[True, True, False], inplace=True)
            labels.drop_duplicates(subset=['symbol', 'entry_time'], keep='first', inplace=True)
            print(f"Объединены метки бота и эксперта. Всего меток: {len(labels)}")
        except Exception as e:
            print(f"Ошибка при загрузке expert_trades.csv: {e}. Используем только метки бота.")
            labels = labels_bot.copy() # Если ошибка, используем только метки бота
    else:
        print("Файл expert_trades.csv не найден или пуст. Используем только метки бота.")
        labels = labels_bot.copy()


    # 3) Загрузим полную историю с индикаторами
    full_history = pd.read_csv('full_history_with_indicators.csv', dtype={'symbol': str})
    # Преобразование в datetime и приведение к UTC
    full_history['timestamp'] = pd.to_datetime(full_history['timestamp'], errors='coerce', utc=True)
    full_history.dropna(subset=['timestamp'], inplace=True) # Удаляем строки с некорректными датами
    full_history.set_index('timestamp', inplace=True) # Устанавливаем timestamp как индекс
    full_history.sort_index(inplace=True) # Сортируем по индексу
    print(f"Загружено {len(full_history)} строк полной истории.")

    print("Объединение размеченных сделок с историческими данными...")

    # Список для хранения результатов
    training_data_rows = []

    # 4) Для каждой размеченной сделки извлекаем данные индикаторов
    all_symbols = labels['symbol'].unique()
    for sym in all_symbols:
        labels_sym = labels[labels['symbol'] == sym].sort_values('entry_time') # Сортируем по entry_time
        history_sym = full_history[full_history['symbol'] == sym].copy()
        
        if history_sym.empty:
            print(f"Нет истории для символа {sym}. Пропускаем.")
            continue

        # Создаем временный индекс с уникальными метками времени для join
        # Это будет `timestamp` из `full_history`
        history_sym.reset_index(inplace=True) # Превращаем индекс обратно в колонку
        
        # Переименовываем колонку 'timestamp' для удобства при объединении
        history_sym.rename(columns={'timestamp': 'join_time'}, inplace=True)
        
        # Важно: ensure `join_time` is timezone-aware if `entry_time` is.
        # Поскольку мы привели entry_time к UTC, нужно убедиться, что join_time тоже UTC.
        # Это уже должно быть так, если full_history['timestamp'] был UTC.
        
        # Делаем merge по ближайшей временной метке
        # Используем merge_asof для поиска ближайшей свечи
        # Sort both dataframes by their time columns before merge_asof
        labels_sym = labels_sym.sort_values('entry_time')
        history_sym = history_sym.sort_values('join_time')

        # Объединяем, ища ближайшую свечу ПЕРЕД entry_time
        merged_df = pd.merge_asof(
            labels_sym,
            history_sym,
            left_on='entry_time',
            right_on='join_time',
            direction='backward', # Ищем свечу, которая закончилась до entry_time
            by='symbol',
            tolerance=pd.Timedelta('10s') # Небольшой допуск, чтобы найти свечу
        )
        
        # Отфильтровываем случаи, когда не удалось найти соответствующую свечу (join_time is NaT)
        merged_df.dropna(subset=['join_time'], inplace=True)

        # Добавляем данные в список
        if not merged_df.empty:
            training_data_rows.append(merged_df)

    if not training_data_rows:
        print("Не удалось объединить ни одну размеченную сделку с историческими данными. Выход.")
        # Создаем пустой DataFrame с нужными колонками
        empty_df_cols = ['symbol', 'side', 'entry_time', 'entry_price', 'exit_time', 'exit_price', 'reason', 'pnl', 'label'] + [col for col in full_history.columns if col not in ['symbol', 'join_time']]
        pd.DataFrame(columns=empty_df_cols).to_csv('training_dataset.csv', index=False)
        return

    combined_df = pd.concat(training_data_rows, ignore_index=True)
    print(f"Объединено {len(combined_df)} строк для обучения.")

    # 5) Создаем признаки
    # Исключаем колонки, которые не являются признаками или целевой переменной
    # 'entry_time', 'exit_time', 'join_time' теперь не нужны в X
    columns_to_exclude = [
        'entry_time', 'exit_time', 'entry_price', 'exit_price',
        'reason', 'pnl', 'source', 'open', 'high', 'low', 'close', 'volume', # Исключаем OHLCV, если уже есть индикаторы
        'symbol', 'join_time', 'side' # <-- ДОБАВЛЕНА 'side'
    ]
    
    # Сохраняем entry_time, symbol и side отдельно для последующего объединения с X и y
    entry_times_to_save = combined_df['entry_time']
    symbols_to_save = combined_df['symbol']
    sides_to_save = combined_df['side'] # <-- ДОБАВЛЕНА ЭТА СТРОКА

    # Удаляем колонки, которые не нужны для признаков, а также целевую переменную 'label'
    X = combined_df.drop(columns=[col for col in columns_to_exclude if col in combined_df.columns] + ['label'], errors='ignore')
    y = combined_df['label']

    # 6) Обрабатываем бесконечные значения и большие числа
    print(f"До обработки inf/NaN: {len(X)} строк.")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 7) Удаляем признаки с большим количеством NaN (если необходимо)
    initial_features_count = X.shape[1]
    nan_threshold = 0.8 # Если более 80% значений NaN
    cols_to_drop_nan = X.columns[X.isnull().sum() / len(X) > nan_threshold].tolist()
    if cols_to_drop_nan:
        X.drop(columns=cols_to_drop_nan, inplace=True)
        print(f"Удалены признаки с более чем {nan_threshold*100}% NaN: {cols_to_drop_nan}")
    print(f"После удаления признаков с большим количеством NaN: {X.shape[1]} признаков.")

    # 8) Заполняем оставшиеся NaN медианой
    # Важно: медиана должна быть рассчитана на обучающем наборе, но для простоты здесь заполняем глобально
    # Для продакшена лучше использовать SimpleImputer в пайплайне
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            if not pd.isna(median_val): # Проверяем, что медиана не NaN (случай, когда колонка полностью NaN)
                X[col].fillna(median_val, inplace=True)
            else:
                # Если колонка полностью NaN, заполняем 0 или удаляем
                print(f"Предупреждение: Колонка '{col}' полностью NaN после фильтрации. Заполнение 0.")
                X[col].fillna(0, inplace=True) # Заполняем 0, если вся колонка NaN
                
    print(f"После заполнения NaN: {len(X)} строк.")

    # 9) Финальная очистка от любых оставшихся NaN/inf (должны быть уже обработаны)
    before_dropna_final = len(X)
    # Создаем DataFrame для final dropna, включая y
    combined_for_dropna = pd.concat([X, y], axis=1) # Используем y как целевую переменную
    combined_for_dropna = combined_for_dropna.dropna()
    print(f"После final dropna: {before_dropna_final} -> {len(combined_for_dropna)} строк.")

    # Разделяем обратно X и y после очистки
    y = combined_for_dropna.pop('label') # Извлекаем метку
    X = combined_for_dropna # Оставшееся - это признаки

    # Синхронизируем entry_times_to_save, symbols_to_save и sides_to_save с отфильтрованными индексами
    # Используем .loc[X.index] для правильной синхронизации
    entry_times_to_save = entry_times_to_save.loc[X.index]
    symbols_to_save = symbols_to_save.loc[X.index]
    sides_to_save = sides_to_save.loc[X.index] # <-- ДОБАВЛЕНА ЭТА СТРОКА

    # Объединяем X, y, entry_time, symbol и side в один DataFrame для сохранения
    training_df = pd.concat([X, y, entry_times_to_save, symbols_to_save, sides_to_save], axis=1) # <-- ОБНОВЛЕНА ЭТА СТРОКА

    # 10) Сохраняем итоговый файл
    output_training_file = 'training_dataset.csv'
    if not training_df.empty:
        training_df.to_csv(output_training_file, index=False)
        print(f"Обучающий датасет сохранен в {output_training_file}. Всего: {len(training_df)} строк.")
        print(f"Распределение меток: \n{training_df['label'].value_counts()}")
        print(f"Использованные признаки (X): {X.columns.tolist()}")
    else:
        # Если DataFrame пуст, создаем пустой файл с заголовками
        print(f"Обучающий датасет пуст. Создаю пустой файл {output_training_file}.")
        # Определяем все возможные колонки, которые должны быть в training_dataset.csv
        all_cols = X.columns.tolist() + ['label', 'entry_time', 'symbol', 'side'] # <-- ОБНОВЛЕНА ЭТА СТРОКА
        pd.DataFrame(columns=all_cols).to_csv(output_training_file, index=False)

if __name__ == '__main__':
    main() 
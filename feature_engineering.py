#!/usr/bin/env python3
# feature_engineering.py

import pandas as pd
import os # Добавляем импорт os для проверки существования файла

def main():
    # 1) Загрузим размеченные сделки от бота
    labels_bot = pd.read_csv('labeled_trades.csv', dtype={'symbol': str})
    labels_bot['entry_time'] = pd.to_datetime(
    labels_bot['entry_time']
)
    labels_bot['exit_time'] = pd.to_datetime(
    labels_bot['exit_time'],
    )
    labels_bot['source'] = 'bot' # Добавляем столбец для отслеживания источника

    # 2) Загрузим экспертные сделки, если файл существует
    expert_trades_file = 'expert_trades.csv'
    if os.path.exists(expert_trades_file):
        try:
            labels_expert = pd.read_csv(expert_trades_file, dtype={'symbol': str})
            labels_expert['entry_time'] = pd.to_datetime(labels_expert['entry_time'])
            labels_expert['exit_time'] = pd.to_datetime(labels_expert['exit_time'])
            labels_expert['source'] = 'expert' # Добавляем столбец для отслеживания источника

            # Объединяем сделки бота и экспертные сделки
            # Если есть пересечения (одинаковый символ и время входа),
            # экспертная разметка будет иметь приоритет (keep='first' после сортировки по источнику)
            labels = pd.concat([labels_bot, labels_expert], ignore_index=True)
            
            # Сортируем так, чтобы 'expert' записи были "первыми" для drop_duplicates
            labels.sort_values(by=['symbol', 'entry_time', 'source'], ascending=[True, True, False], inplace=True)
            # Удаляем дубликаты, сохраняя экспертную запись, если она совпадает со сделкой бота
            labels.drop_duplicates(subset=['symbol', 'entry_time'], keep='first', inplace=True)
            
            print(f"Загружено {len(labels_bot)} сделок бота и {len(labels_expert)} экспертных сделок.")
            print(f"Итого после объединения и удаления дубликатов: {len(labels)} уникальных сделок.")

        except pd.errors.EmptyDataError:
            print(f"Файл {expert_trades_file} пуст. Использую только сделки бота.")
            labels = labels_bot
        except Exception as e:
            print(f"Ошибка при загрузке {expert_trades_file}: {e}. Использую только сделки бота.")
            labels = labels_bot
    else:
        print(f"Файл {expert_trades_file} не найден. Использую только сделки бота.")
        labels = labels_bot

    # Удаляем столбец 'source', так как он больше не нужен для обучения модели
    labels.drop(columns=['source'], errors='ignore', inplace=True)


    # 3) Загрузим историю с индикаторами
    hist = pd.read_csv('full_history_with_indicators.csv', dtype={'symbol': str})

    try:
        hist['timestamp'] = pd.to_datetime(
            hist['timestamp'],
            format='%Y-%m-%d %H:%M:%S.%f',
            exact=False
        )
    except ValueError:
        hist['timestamp'] = pd.to_datetime(
            hist['timestamp'],
        )

    # 4) Сортируем по символу и времени (снова, после возможного объединения)
    labels = labels.sort_values(['symbol', 'entry_time']).reset_index(drop=True)
    hist   = hist.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

    # 5) По-символьный merge_asof с tolerance в 1 минуту
    merged_parts = []
    for sym in labels['symbol'].unique():
        lb = labels[labels['symbol'] == sym]
        hi = hist[hist['symbol']    == sym]
        m = pd.merge_asof(
            left=lb,
            right=hi,
            left_on='entry_time',
            right_on='timestamp',
            by='symbol',
            direction='backward',
            tolerance=pd.Timedelta('1m')
        )
        merged_parts.append(m)
    data = pd.concat(merged_parts, ignore_index=True)

    # 6) Удаляем строки, где не нашлось соответствия в истории
    initial_rows = len(data)
    data.dropna(subset=['timestamp'], inplace=True) # Если 'timestamp' NaN, значит merge_asof не нашел соответствия
    print(f"Удалено {initial_rows - len(data)} строк без соответствующей исторической свечи.")


    # 7) Выбираем признаки и метку
    feature_cols = [
        'ATR_14',
        'EMA_50', 'EMA_200',
        'RSI_14',
        'MACD_12_26_9', 'MACDs_12_26_9',
        'BB_middle', 'BB_width',
        'VOLUME_EMA',
        'KAMA_10_2_30',
        'CMF_20',
        'RVI_14',
        'volume'
    ]

    # Убедимся, что все feature_cols присутствуют в данных
    missing_features = [col for col in feature_cols if col not in data.columns]
    if missing_features:
        raise ValueError(f"Отсутствуют необходимые признаки в full_history_with_indicators.csv: {missing_features}. Проверьте generate_history.py")

    X = data[feature_cols].copy()
    y = data["label"].copy()

    # Объединяем X и y для сохранения в один CSV
    training_dataset = pd.concat([X, y], axis=1)

    # 8) Сохраняем обучающий датасет
    training_dataset.to_csv("training_dataset.csv", index=False)
    print(f"Обучающий датасет сохранен в training_dataset.csv. Размер: {len(training_dataset)} строк.")

if __name__ == "__main__":
    main()
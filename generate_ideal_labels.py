import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# --- Параметры для определения "идеальных" сделок ---
CONFIG_FILE = 'config.json'
HISTORY_FILE = 'full_history_with_indicators.csv'
OUTPUT_FILE = 'expert_trades.csv'

# Параметры поиска идеальной сделки (для label=1)
# Целевой процент прибыли
TARGET_PROFIT_PERCENT = 0.06 # Ищем сделки, которые принесли бы 2% прибыли
# Максимальный процент убытка, который мы готовы пережить ДО достижения прибыли
MAX_DRAWDOWN_PERCENT   = 0.05 # Не более 2% просадки перед достижением прибыли
# Окно поиска идеальной сделки (количество свечей вперед)
LOOKAHEAD_CANDLES = 300 # Смотрим на следующие 40 свечей

# Параметры для генерации меток 0 (неидеальные/убыточные)
# Если сделка не является идеальной (label=1)
# И ее PnL в конце окна LOOKAHEAD_CANDLES меньше или равен этому порогу,
# то она будет помечена как label=0.
# Это помогает включить в 0-класс как убыточные, так и очень низкодоходные сделки.
ZERO_LABEL_PNL_THRESHOLD = 0.00 # Если PnL <= 0%, то это метка 0

def generate_labels(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Генерирует метки LONG и SHORT сделок (1 для идеальных, 0 для неидеальных/убыточных),
    анализируя будущее движение цены.
    """
    records = []

    # Проходим по всем свечам, кроме последних LOOKAHEAD_CANDLES, так как для них
    # невозможно сформировать полное окно будущего
    for i in range(len(df) - LOOKAHEAD_CANDLES):
        current_candle = df.iloc[i]
        # Окно будущих свечей, начиная со следующей
        future_window = df.iloc[i+1 : i + 1 + LOOKAHEAD_CANDLES]

        if future_window.empty:
            continue

        current_close = current_candle['close']
        
        # --- Анализ для LONG сделок ---
        max_price_in_window_long = future_window['high'].max()
        min_price_in_window_long = future_window['low'].min()

        # Расчет потенциальной прибыли и просадки для LONG
        pnl_long = (max_price_in_window_long - current_close) / current_close
        drawdown_long = (current_close - min_price_in_window_long) / current_close

        # Определяем метку для LONG
        label_long = 0
        reason_long = 'NON_IDEAL_LONG_LOSS_OR_FLAT'
        # По умолчанию, если не идеальная, то PnL - это PnL в конце окна
        pnl_for_record_long = (future_window.iloc[-1]['close'] - current_close) / current_close

        # Если условия для идеальной LONG выполнены
        if (pnl_long >= TARGET_PROFIT_PERCENT and drawdown_long <= MAX_DRAWDOWN_PERCENT):
            label_long = 1
            reason_long = 'IDEAL_PROFIT_LONG'
            pnl_for_record_long = pnl_long # Используем идеальный PnL для записи

        # Добавляем запись, если это идеальная сделка (label=1) ИЛИ
        # если это неидеальная сделка (label=0) с PnL ниже порога
        if label_long == 1 or pnl_for_record_long <= ZERO_LABEL_PNL_THRESHOLD:
            records.append({
                'symbol':       symbol, 'side': 'LONG',
                'entry_time':   current_candle['timestamp'], 'entry_price': current_close,
                'exit_time':    future_window.iloc[-1]['timestamp'], # Время выхода: конец окна для 0, или когда достигнута цель для 1 (но мы берем конец окна для простоты)
                'exit_price':   future_window.iloc[-1]['close'], # Цена выхода: цена на конец окна
                'reason':       reason_long, 'pnl': pnl_for_record_long, 'label': label_long
            })


        # --- Анализ для SHORT сделок ---
        max_price_in_window_short = future_window['high'].max()
        min_price_in_window_short = future_window['low'].min()

        # Расчет потенциальной прибыли и просадки для SHORT
        pnl_short = (current_close - min_price_in_window_short) / current_close
        drawdown_short = (max_price_in_window_short - current_close) / current_close

        # Определяем метку для SHORT
        label_short = 0
        reason_short = 'NON_IDEAL_SHORT_LOSS_OR_FLAT'
        # По умолчанию, если не идеальная, то PnL - это PnL в конце окна
        pnl_for_record_short = (current_close - future_window.iloc[-1]['close']) / current_close

        # Если условия для идеальной SHORT выполнены
        if (pnl_short >= TARGET_PROFIT_PERCENT and drawdown_short <= MAX_DRAWDOWN_PERCENT):
            label_short = 1
            reason_short = 'IDEAL_PROFIT_SHORT'
            pnl_for_record_short = pnl_short # Используем идеальный PnL для записи

        # Добавляем запись, если это идеальная сделка (label=1) ИЛИ
        # если это неидеальная сделка (label=0) с PnL ниже порога
        if label_short == 1 or pnl_for_record_short <= ZERO_LABEL_PNL_THRESHOLD:
            records.append({
                'symbol':       symbol, 'side': 'SHORT',
                'entry_time':   current_candle['timestamp'], 'entry_price': current_close,
                'exit_time':    future_window.iloc[-1]['timestamp'], # Время выхода: конец окна для 0, или когда достигнута цель для 1
                'exit_price':   future_window.iloc[-1]['close'], # Цена выхода: цена на конец окна
                'reason':       reason_short, 'pnl': pnl_for_record_short, 'label': label_short
            })

    return pd.DataFrame(records)


def main():
    # 1) Загрузить конфиг для SYMBOLS
    with open(CONFIG_FILE, encoding='utf-8') as f:
        cfg = json.load(f)
    SYMBOLS = cfg['symbols']

    # 2) Загрузить полную историю с индикаторами
    print(f"Загрузка истории из {HISTORY_FILE}...")
    try:
        hist = pd.read_csv(HISTORY_FILE, dtype={'symbol': str})
        hist['timestamp'] = pd.to_datetime(hist['timestamp'])
        print(f"Загружено {len(hist)} строк истории.")
    except FileNotFoundError:
        print(f"Ошибка: файл {HISTORY_FILE} не найден. Сначала запустите generate_history.py.")
        return
    
    all_generated_labels = []

    # 3) Генерируем новые метки для каждого символа
    for sym in SYMBOLS:
        print(f"→ Генерация меток для {sym}")
        sym_df = hist[hist['symbol'] == sym].sort_values('timestamp').reset_index(drop=True)
        if sym_df.empty:
            print(f"   Нет данных для {sym}, пропуск.")
            continue
        
        generated_labels_sym = generate_labels(sym_df, sym)
        if not generated_labels_sym.empty:
            all_generated_labels.append(generated_labels_sym)
            print(f"   Сгенерировано {len(generated_labels_sym)} новых меток для {sym} (вкл. 0 и 1)")
        else:
            print(f"   Не найдено новых меток для {sym} с текущими параметрами.")

    # 4) Объединяем все новые сгенерированные метки
    newly_generated_df = pd.DataFrame()
    if all_generated_labels:
        newly_generated_df = pd.concat(all_generated_labels, ignore_index=True)
        # Убедимся, что столбцы соответствуют ожиданиям
        required_cols = [
            'symbol', 'side', 'entry_time', 'entry_price',
            'exit_time', 'exit_price', 'reason', 'pnl', 'label'
        ]
        newly_generated_df = newly_generated_df[required_cols]
    
    # 5) Загружаем существующие метки из файла, если он есть
    existing_df = pd.DataFrame(columns=required_cols) # Пустой DataFrame для начала
    if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
        try:
            existing_df = pd.read_csv(OUTPUT_FILE, dtype={'symbol': str})
            # Конвертируем столбцы времени в datetime, если они существуют
            if 'entry_time' in existing_df.columns:
                existing_df['entry_time'] = pd.to_datetime(existing_df['entry_time'])
            if 'exit_time' in existing_df.columns:
                existing_df['exit_time'] = pd.to_datetime(existing_df['exit_time'])
            print(f"Загружено {len(existing_df)} существующих меток из {OUTPUT_FILE}.")
        except pd.errors.EmptyDataError:
            print(f"Файл {OUTPUT_FILE} существует, но пуст.")
        except Exception as e:
            print(f"Ошибка при загрузке {OUTPUT_FILE}: {e}. Продолжаем без старых меток.")

    # 6) Объединяем существующие и новые метки
    # Используем pd.concat даже если один из DataFrame пуст
    combined_df = pd.concat([existing_df, newly_generated_df], ignore_index=True)

    # 7) Удаляем дубликаты
    # Дубликаты определяются по комбинации символа, времени входа и стороне сделки
    # 'keep='first'' сохраняет первую встреченную запись (т.е. существующую, если она была)
    initial_len = len(combined_df)
    final_unique_df = combined_df.drop_duplicates(
        subset=['symbol', 'entry_time', 'side'],
        keep='first'
    )
    if initial_len > len(final_unique_df):
        print(f"Удалено {initial_len - len(final_unique_df)} дубликатов.")

    # 8) Сохраняем итоговый файл
    if not final_unique_df.empty:
        final_unique_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Все уникальные сгенерированные метки сохранены в {OUTPUT_FILE}. Всего: {len(final_unique_df)} меток.")
        print(f"Распределение меток: \n{final_unique_df['label'].value_counts()}")
    else:
        # Если в итоге нет меток, убедимся, что файл пуст
        pd.DataFrame(columns=required_cols).to_csv(OUTPUT_FILE, index=False)
        print(f"Не найдено меток для сохранения. Создан или обновлен пустой файл {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()
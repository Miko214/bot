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
TARGET_PROFIT_PERCENT = 0.02 # Ищем сделки, которые принесли бы 2% прибыли
# Максимальный процент убытка, который мы готовы пережить ДО достижения прибыли
MAX_DRAWDOWN_PERCENT   = 0.02 # Не более 2% просадки перед достижением прибыли
# Добавляем параметр стоп-лосса
STOP_LOSS_PERCENT      = 0.02 # Если цена двинулась против нас на 2%, считаем это стоп-лоссом
# Окно поиска идеальной сделки (количество свечей вперед)
LOOKAHEAD_CANDLES = 100 # Смотрим на следующие 40 свечей

# Параметры для генерации меток 0 (неидеальные/убыточные)
# Если сделка не является идеальной (label=1)
# И ее PnL в конце окна LOOKAHEAD_CANDLES меньше или равен этому порогу,
# то она будет помечена как label=0.
# Это помогает включить в 0-класс как убыточные, так и очень низкодоходные сделки.
ZERO_LABEL_PNL_THRESHOLD = 0.00 # Если PnL <= 0%, то это метка 0

def generate_labels(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Генерирует метки LONG и SHORT сделок (1 для идеальных, 0 для неидеальных/убыточных),
    анализируя будущее движение цены с учетом TP/SL.
    """
    records = []

    # Проходим по всем свечам, кроме последних LOOKAHEAD_CANDLES
    for i in range(len(df) - LOOKAHEAD_CANDLES):
        current_candle = df.iloc[i]
        # Окно будущих свечей, начиная со следующей
        future_window = df.iloc[i+1 : i + 1 + LOOKAHEAD_CANDLES]

        if future_window.empty:
            continue

        current_close = current_candle['close']
        
        # --- Анализ для LONG сделок ---
        target_profit_price_long = current_close * (1 + TARGET_PROFIT_PERCENT)
        stop_loss_price_long     = current_close * (1 - STOP_LOSS_PERCENT)
        
        # Переменные для отслеживания результатов в окне
        reached_profit_long = False
        reached_stop_loss_long = False
        exit_price_long = current_close # Инициализируем на случай, если ни TP, ни SL не достигнуты
        exit_time_long = current_candle['timestamp']
        pnl_at_exit_long = 0.0

        for idx, future_c in future_window.iterrows():
            # Проверяем на достижение TP
            if future_c['high'] >= target_profit_price_long:
                reached_profit_long = True
                exit_price_long = target_profit_price_long
                exit_time_long = future_c['timestamp'] # Время свечи, когда достигли
                pnl_at_exit_long = TARGET_PROFIT_PERCENT
                break # Выходим, так как цель достигнута

            # Проверяем на достижение SL (только если TP не был достигнут раньше в этой же свече)
            if future_c['low'] <= stop_loss_price_long:
                # Если TP и SL в одной свече, приоритет TP (можно изменить)
                if not reached_profit_long:
                    reached_stop_loss_long = True
                    exit_price_long = stop_loss_price_long
                    exit_time_long = future_c['timestamp']
                    pnl_at_exit_long = -STOP_LOSS_PERCENT
                    break # Выходим, так как стоп-лосс достигнут
        
        label_long = 0
        reason_long = 'NON_IDEAL_LONG_LOSS_OR_FLAT' # По умолчанию
        
        if reached_profit_long:
            label_long = 1
            reason_long = 'IDEAL_PROFIT_LONG_TP'
        elif reached_stop_loss_long:
            label_long = 0
            reason_long = 'NON_IDEAL_LONG_SL_HIT'
            # PnL уже установлен в -STOP_LOSS_PERCENT
        else:
            # Если ни TP, ни SL не были достигнуты, смотрим PnL в конце окна
            final_close_in_window = future_window.iloc[-1]['close']
            pnl_at_exit_long = (final_close_in_window - current_close) / current_close
            exit_price_long = final_close_in_window
            exit_time_long = future_window.iloc[-1]['timestamp'] # Время последней свечи окна

            # Проверяем условие для метки 0 на основе PnL в конце окна
            if pnl_at_exit_long <= ZERO_LABEL_PNL_THRESHOLD:
                label_long = 0
                reason_long = 'NON_IDEAL_LONG_LOSS_OR_FLAT'
            # else: # Если PnL > ZERO_LABEL_PNL_THRESHOLD, но не достиг TP, то эта сделка не будет записана
            #     continue # Пропускаем, так как не 1 и не 0 по условию

        # Добавляем запись, если это идеальная сделка (label=1) ИЛИ
        # если это неидеальная сделка (label=0) с PnL ниже порога
        # или если это сделка, закрытая по SL
        if label_long == 1 or label_long == 0 and (pnl_at_exit_long <= ZERO_LABEL_PNL_THRESHOLD or reason_long == 'NON_IDEAL_LONG_SL_HIT'):
            records.append({
                'symbol':       symbol, 'side': 'LONG',
                'entry_time':   current_candle['timestamp'], 'entry_price': current_close,
                'exit_time':    exit_time_long,
                'exit_price':   exit_price_long,
                'reason':       reason_long, 'pnl': pnl_at_exit_long, 'label': label_long
            })


        # --- Анализ для SHORT сделок ---
        target_profit_price_short = current_close * (1 - TARGET_PROFIT_PERCENT)
        stop_loss_price_short     = current_close * (1 + STOP_LOSS_PERCENT)
        
        reached_profit_short = False
        reached_stop_loss_short = False
        exit_price_short = current_close
        exit_time_short = current_candle['timestamp']
        pnl_at_exit_short = 0.0

        for idx, future_c in future_window.iterrows():
            # Проверяем на достижение TP
            if future_c['low'] <= target_profit_price_short:
                reached_profit_short = True
                exit_price_short = target_profit_price_short
                exit_time_short = future_c['timestamp']
                pnl_at_exit_short = TARGET_PROFIT_PERCENT
                break

            # Проверяем на достижение SL
            if future_c['high'] >= stop_loss_price_short:
                if not reached_profit_short:
                    reached_stop_loss_short = True
                    exit_price_short = stop_loss_price_short
                    exit_time_short = future_c['timestamp']
                    pnl_at_exit_short = -STOP_LOSS_PERCENT
                    break
        
        label_short = 0
        reason_short = 'NON_IDEAL_SHORT_LOSS_OR_FLAT'

        if reached_profit_short:
            label_short = 1
            reason_short = 'IDEAL_PROFIT_SHORT_TP'
        elif reached_stop_loss_short:
            label_short = 0
            reason_short = 'NON_IDEAL_SHORT_SL_HIT'
        else:
            final_close_in_window = future_window.iloc[-1]['close']
            pnl_at_exit_short = (current_close - final_close_in_window) / current_close
            exit_price_short = final_close_in_window
            exit_time_short = future_window.iloc[-1]['timestamp']

            if pnl_at_exit_short <= ZERO_LABEL_PNL_THRESHOLD:
                label_short = 0
                reason_short = 'NON_IDEAL_SHORT_LOSS_OR_FLAT'
            # else:
            #     continue

        if label_short == 1 or label_short == 0 and (pnl_at_exit_short <= ZERO_LABEL_PNL_THRESHOLD or reason_short == 'NON_IDEAL_SHORT_SL_HIT'):
            records.append({
                'symbol':       symbol, 'side': 'SHORT',
                'entry_time':   current_candle['timestamp'], 'entry_price': current_close,
                'exit_time':    exit_time_short,
                'exit_price':   exit_price_short,
                'reason':       reason_short, 'pnl': pnl_at_exit_short, 'label': label_short
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
        # Используем .reindex() для гарантии порядка столбцов, если их порядок мог измениться
        # и чтобы добавить отсутствующие столбцы с NaN, если это необходимо
        newly_generated_df = newly_generated_df.reindex(columns=required_cols)
    
    final_unique_df = newly_generated_df 


    # 5) Сохраняем итоговый файл
    if not final_unique_df.empty:
        final_unique_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Все сгенерированные метки перезаписаны в {OUTPUT_FILE}. Всего: {len(final_unique_df)} меток.")
        print(f"Распределение меток: \n{final_unique_df['label'].value_counts()}")
    else:
        # Если в итоге нет меток, убедимся, что файл пуст (или создаем его пустым)
        # Убедитесь, что required_cols определен до этого блока
        required_cols = [
            'symbol', 'side', 'entry_time', 'entry_price',
            'exit_time', 'exit_price', 'reason', 'pnl', 'label'
        ] # Добавлено для надежности, если newly_generated_df пуст
        pd.DataFrame(columns=required_cols).to_csv(OUTPUT_FILE, index=False)
        print(f"Не найдено меток для сохранения. Создан или перезаписан пустой файл {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()
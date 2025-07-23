import json
import ccxt
import pandas as pd
import pandas_ta as ta
import warnings

# Подавляем предупреждение о pkg_resources
warnings.filterwarnings('ignore', category=UserWarning, module='pandas_ta')

# --- Параметры ---
CONFIG_FILE = 'config.json'
OUTPUT_FILE = 'full_history_with_indicators.csv'
HISTORY_LIMIT = 5000

# Настройки индикаторов
ATR_LENGTH = 14
RSI_LENGTH = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_LENGTH = 20
BB_STD = 2.0
EMA_SHORT = 50
EMA_LONG = 200
VOLUME_EMA_LENGTH = 20
KAMA_LENGTH = 10
KAMA_FAST_EMA = 2
KAMA_SLOW_EMA = 30
CMF_LENGTH = 20
RVI_LENGTH = 14
STOCH_LENGTH = 14
STOCH_SMOOTH_K = 3
STOCH_SMOOTH_D = 3
ADX_LENGTH = 14
CCI_LENGTH = 20
VWAP_LENGTH = 14

# 1) Загрузить конфиг
with open(CONFIG_FILE, encoding='utf-8') as f:
    cfg = json.load(f)
SYMBOLS = cfg['symbols']
TIMEFRAME = cfg['timeframe']

# 2) Инициализация биржи
exchange = ccxt.binance({
    'rateLimit': 1200,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'adjustForTimeDifference': True, # Учитывать разницу во времени
    },
})


def fetch_ohlcv(symbol: str) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=HISTORY_LIMIT)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Добавляем временные признаки
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month_of_year'] = df.index.month
    df['is_weekend'] = ((df.index.dayofweek == 5) | (df.index.dayofweek == 6)).astype(int)

    # ATR
    df.ta.atr(length=ATR_LENGTH, append=True)

    # EMA
    df.ta.ema(length=EMA_SHORT, append=True)
    df.ta.ema(length=EMA_LONG, append=True)

    # RSI
    df.ta.rsi(length=RSI_LENGTH, append=True)

    # MACD
    df.ta.macd(
        fast=MACD_FAST,
        slow=MACD_SLOW,
        signal=MACD_SIGNAL,
        append=True
    )

    # Bollinger Bands
    df.ta.bbands(
        close=df['close'],
        length=BB_LENGTH,
        std=BB_STD,
        append=True
    )
    # Переименовываем Bollinger Bands колонки
    if f'BBL_{BB_LENGTH}_{BB_STD}' in df.columns:
        df.rename(columns={f'BBL_{BB_LENGTH}_{BB_STD}': 'BB_lower'}, inplace=True)
    if f'BBM_{BB_LENGTH}_{BB_STD}' in df.columns:
        df.rename(columns={f'BBM_{BB_LENGTH}_{BB_STD}': 'BB_middle'}, inplace=True)
    if f'BBU_{BB_LENGTH}_{BB_STD}' in df.columns:
        df.rename(columns={f'BBU_{BB_LENGTH}_{BB_STD}': 'BB_upper'}, inplace=True)
    if f'BBB_{BB_LENGTH}_{BB_STD}' in df.columns:  # BBB - это Bandwidth, который мы назвали BB_width
        df.rename(columns={f'BBB_{BB_LENGTH}_{BB_STD}': 'BB_width'}, inplace=True)


    # VOLUME_EMA (для объёма)
    df.ta.ema(close=df['volume'], length=VOLUME_EMA_LENGTH, append=True, col_names=(f'VOLUME_EMA',))

    # KAMA
    df.ta.kama(
        close=df['close'],
        length=KAMA_LENGTH,
        fast=KAMA_FAST_EMA,
        slow=KAMA_SLOW_EMA,
        append=True
    )

    # CMF
    df.ta.cmf(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        volume=df['volume'],
        length=CMF_LENGTH,
        append=True
    )

    # RVI
    df.ta.rvi(close=df['close'], length=RVI_LENGTH, append=True)

    # Stochastic Oscillator
    df.ta.stoch(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        k=STOCH_LENGTH,
        d=STOCH_SMOOTH_D,
        append=True
    )
    # Переименовываем Stochastic колонки
    if f'STOCHk_{STOCH_LENGTH}_{STOCH_SMOOTH_K}_{STOCH_SMOOTH_D}' in df.columns:
        df.rename(columns={f'STOCHk_{STOCH_LENGTH}_{STOCH_SMOOTH_K}_{STOCH_SMOOTH_D}': 'STOCH_k'}, inplace=True)
    if f'STOCHd_{STOCH_LENGTH}_{STOCH_SMOOTH_K}_{STOCH_SMOOTH_D}' in df.columns:
        df.rename(columns={f'STOCHd_{STOCH_LENGTH}_{STOCH_SMOOTH_K}_{STOCH_SMOOTH_D}': 'STOCH_d'}, inplace=True)


    # ADX
    df.ta.adx(length=ADX_LENGTH, append=True)

    # CCI
    df.ta.cci(length=CCI_LENGTH, append=True)

    # VWAP (если есть volume)
    if 'volume' in df.columns and not df['volume'].isnull().all():
        df.ta.vwap(append=True, fillna=True) # VWAP обычно не принимает length

    # Удаляем строки с NaN значениями, которые появились из-за индикаторов
    # Список необходимых колонок для обучения
    required = [
        f'ATR_{ATR_LENGTH}',
        f'EMA_{EMA_SHORT}', f'EMA_{EMA_LONG}',
        f'RSI_{RSI_LENGTH}',
        f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}',
        f'MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}',
        f'MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}',
        'BB_upper', 'BB_middle', 'BB_lower', 'BB_width', # ОБНОВЛЕНО
        'VOLUME_EMA',
        f'KAMA_{KAMA_LENGTH}_{KAMA_FAST_EMA}_{KAMA_SLOW_EMA}',
        f'CMF_{CMF_LENGTH}',
        f'RVI_{RVI_LENGTH}',
        'STOCH_k', 'STOCH_d', # ОБНОВЛЕНО
        f'ADX_{ADX_LENGTH}',
        f'CCI_{CCI_LENGTH}'
    ]
    # Добавляем VWAP только если он успешно вычислен
    if f'VWAP_{VWAP_LENGTH}' in df.columns and not df[f'VWAP_{VWAP_LENGTH}'].isna().all():
        required.append(f'VWAP_{VWAP_LENGTH}')

    # Фильтруем required до фактически существующих колонок после индикаторов и переименований
    required = [col for col in required if col in df.columns]
    df.dropna(subset=required, inplace=True)

    return df


def main():
    all_dfs = []

    for sym in SYMBOLS:
        print(f"→ Fetching {sym}")
        df = fetch_ohlcv(sym) # Предполагается, что fetch_ohlcv уже возвращает UTC datetime
        if df.empty:
            print("   skipped — no data")
            continue

        df_ind = compute_all_indicators(df)
        if df_ind.empty:
            print("   skipped — indicators could not be computed")
            continue

        df_ind['symbol'] = sym  # Добавляем столбец с символом
        # Добавляем признак 'hour_of_day' из индекса (который должен быть timestamp)
        # Если timestamp уже DatetimeIndex в UTC, то .hour будет корректным.
        df_ind['hour_of_day'] = df_ind.index.hour 
            
        all_dfs.append(df_ind)

    if all_dfs:
        full_history = pd.concat(all_dfs).reset_index()
        # Ensure timestamp is in UTC
        # ИСПРАВЛЕНО: Удален 'errors='coerce'' из tz_localize
        # Если 'timestamp' уже в UTC, эта строка просто переубедится в этом.
        # Если она была наивной, она будет локализована как UTC.
        full_history['timestamp'] = full_history['timestamp'].dt.tz_localize(None).dt.tz_localize('UTC')
        
        # Если 'hour_of_day' не был создан ранее (например, если df_ind.index не был DatetimeIndex)
        # Это дополнительная проверка, если df_ind.index.hour не сработал.
        if 'hour_of_day' not in full_history.columns:
            full_history['hour_of_day'] = full_history['timestamp'].dt.hour
        
        full_history.to_csv(OUTPUT_FILE, index=False)
        print(f"Полная история с индикаторами и признаками сохранена в {OUTPUT_FILE}. Всего: {len(full_history)} строк.")
    else:
        print("Нет данных для сохранения полной истории.")
        # Создаем пустой CSV с заголовками, чтобы избежать ошибок
        empty_df_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'hour_of_day']
        # Здесь вы можете добавить все имена колонок индикаторов, если хотите.
        # Например, возьмите их из `required` списка в `compute_all_indicators`.
        # Для минимального набора:
        pd.DataFrame(columns=empty_df_cols).to_csv(OUTPUT_FILE, index=False)

if __name__ == '__main__':
    main()
import json
import ccxt
import pandas as pd
import pandas_ta as ta
import warnings
import numpy as np

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
ADX_LENGTH = 14

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
    # --- Critical pre-check for OHLCV data ---
    initial_rows = len(df)
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
    if len(df) < initial_rows:
        print(f"DEBUG: Removed {initial_rows - len(df)} rows with NaN in OHLCV data.")
    
    # Если данных слишком мало после очистки для любых индикаторов
    if df.empty:
        print("DEBUG: DataFrame пуст после очистки OHLCV данных. Возврат пустого DataFrame.")
        return pd.DataFrame()

    # --- РУЧНОЙ РАСЧЕТ ATR ---
    atr_col_name = f'ATR_{ATR_LENGTH}'

    # 1. Рассчитываем True Range (TR) для каждой свечи
    # Смещаем 'close' на 1 назад, чтобы получить 'Close_prev'
    df['prev_close'] = df['close'].shift(1)

    # Защита от NaN в первой свече (или после пропусков): prev_close будет NaN.
    # Для первой свечи (где prev_close = NaN), abs(high - prev_close) и abs(low - prev_close) будут NaN.
    # Если ATR должен быть вычислен для n свечей, то первые n-1 свечей будут иметь NaN ATR.
    # Мы заполняем 0, чтобы избежать проблем.
    
    # H - L
    tr1 = df['high'] - df['low']
    # |H - C_prev|
    tr2 = np.abs(df['high'] - df['prev_close'])
    # |L - C_prev|
    tr3 = np.abs(df['low'] - df['prev_close'])

    # True Range - это максимум из этих трех значений
    df['TR'] = np.maximum(np.maximum(tr1, tr2), tr3)

    # 2. Рассчитываем EMA для TR, чтобы получить ATR
    # pandas_ta использует EMA, но мы можем использовать встроенный метод .ewm для экспоненциальной скользящей средней
    # span - это период EMA (равный ATR_LENGTH)
    # adjust=False для совместимости с формулой EMA, используемой многими TA библиотеками
    df[atr_col_name] = df['TR'].ewm(span=ATR_LENGTH, adjust=False).mean()

    # Заполняем NaN в начале ATR нулями.
    # ATR будет NaN для первых `ATR_LENGTH - 1` свечей (если использовать adjust=True)
    # или для первой свечи из-за `prev_close` (если `adjust=False`).
    # Мы хотим, чтобы ATR_14 был числовым для эксперт-сделок.
    df[atr_col_name] = df[atr_col_name].fillna(0.0)

    # Проверка на успешность создания и неполную NaN-ность
    if atr_col_name not in df.columns or df[atr_col_name].isna().all() or (df[atr_col_name] == 0).all():
        print(f"DEBUG: {atr_col_name} все еще не был успешно создан, полностью NaN или все нули после ручного расчета. Это очень странно.")
        # Дополнительная проверка: если все цены одинаковые, ATR будет 0.
        # В этом случае, это корректный ATR. Но если проблема в данных, то нет.
        # В любом случае, если он 0, expert_trades.py будет использовать 0, что равноценно отключению ATR.
    else:
        print(f"DEBUG: {atr_col_name} успешно рассчитан вручную. Первое ненулевое ATR: {df[atr_col_name][df[atr_col_name] != 0].iloc[0] if not df[atr_col_name][df[atr_col_name] != 0].empty else 'N/A'}")

    # Удаляем вспомогательные колонки
    df.drop(columns=['prev_close', 'TR'], inplace=True, errors='ignore')

    # --- Остальные индикаторы (если вы хотите их оставить) ---
    # Если вы хотите, чтобы другие индикаторы продолжали использоваться pandas_ta,
    # убедитесь, что вы импортируете `pandas_ta` и используете `append=True`
    # Если нет, то их тоже нужно либо удалить, либо рассчитать вручную.
    # Для простоты я оставляю их как есть, предполагая, что они работают корректно.

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
    if f'BBB_{BB_LENGTH}_{BB_STD}' in df.columns:  # BBB - Bandwidth
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

    # ADX
    df.ta.adx(length=ADX_LENGTH, append=True)

    # --- REMOVED UNNECESSARY INDICATORS (as per previous versions) ---

    # Drop rows with NaN values resulting from indicator calculations
    required = [
        f'ATR_{ATR_LENGTH}', # Убедимся, что он здесь, если был создан
        f'EMA_{EMA_SHORT}', f'EMA_{EMA_LONG}',
        f'RSI_{RSI_LENGTH}',
        f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}',
        f'MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}',
        f'MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}',
        'BB_upper', 'BB_middle', 'BB_lower', 'BB_width',
        'VOLUME_EMA',
        f'KAMA_{KAMA_LENGTH}_{KAMA_FAST_EMA}_{KAMA_SLOW_EMA}',
        f'CMF_{CMF_LENGTH}',
        f'RVI_{RVI_LENGTH}',
        f'ADX_{ADX_LENGTH}',
    ]
    
    initial_cols = set(df.columns)
    final_required = [col for col in required if col in initial_cols]

    missing_from_expected = [col for col in required if col not in initial_cols]
    if missing_from_expected:
        print(f"⚠️ Warning: The following expected indicators are missing from the DataFrame and will not be used: {missing_from_expected}.")
        
    df.dropna(subset=final_required, inplace=True)
    
    # Fill any remaining NaNs in indicator columns with 0 after the main dropna
    indicator_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    df[indicator_cols] = df[indicator_cols].fillna(0)

    return df


def main():
    all_dfs = []

    for sym in SYMBOLS:
        print(f"→ Fetching {sym}")
        df = fetch_ohlcv(sym)
        if df.empty:
            print(f"   {sym} skipped — no data")
            continue

        df_ind = compute_all_indicators(df)
        if df_ind.empty:
            print(f"   {sym} skipped — indicators could not be computed or all rows dropped.")
            continue

        df_ind['symbol'] = sym  # Добавляем столбец с символом
        all_dfs.append(df_ind)

    if all_dfs:
        full_history = pd.concat(all_dfs).reset_index()
        # Ensure timestamp is in UTC
        if full_history['timestamp'].dt.tz is None:
            full_history['timestamp'] = full_history['timestamp'].dt.tz_localize('UTC')
        else:
            full_history['timestamp'] = full_history['timestamp'].dt.tz_convert('UTC')

        full_history.to_csv(OUTPUT_FILE, index=False)
        print(f"✔ Saved {len(full_history)} rows → {OUTPUT_FILE}")
    else:
        print("❗ No data fetched for any symbol. No history file generated.")


if __name__ == '__main__':
    main()
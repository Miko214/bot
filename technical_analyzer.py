# -*- coding: utf-8 -*-
import ccxt
import pandas as pd
import pandas_ta as ta
import time
# ИСПРАВЛЕНО: Правильный импорт для datetime.strptime
from datetime import datetime
import numpy as np
import logging
import json
import os
import joblib
import glob
import re # Добавлен импорт re для регулярных выражений

import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names*",
    category=UserWarning
)

# --- НОВАЯ ФУНКЦИЯ ДЛЯ ЗАГРУЗКИ ПОСЛЕДНЕЙ МОДЕЛИ (ИСПРАВЛЕНО) ---
def load_latest_model():
    """
    Ищет в папке 'models' файлы trade_model_*.pkl,
    извлекает метку времени из имени файла и загружает самый свежий.
    """
    model_dir = "models"
    pattern = "trade_model_*.pkl"
    files = glob.glob(os.path.join(model_dir, pattern))
    if not files:
        raise FileNotFoundError(f"В папке '{model_dir}' нет ни одного файла '{pattern}'")

    latest_file = None
    latest_timestamp_dt = None # Храним объект datetime для сравнения

    # Регулярное выражение для извлечения метки времени из имени файла
    # Например: trade_model_20250710_1441.pkl -> "20250710_1441"
    timestamp_pattern = re.compile(r"trade_model_(\d{8}_\d{4})\.pkl$")

    for f in files:
        base_name = os.path.basename(f)
        match = timestamp_pattern.search(base_name)
        if match:
            timestamp_str = match.group(1) # Получаем строку метки времени (например, "20250710_1441")
            try:
                # Преобразуем строку в объект datetime для корректного сравнения
                current_timestamp_dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M")
                if latest_timestamp_dt is None or current_timestamp_dt > latest_timestamp_dt:
                    latest_timestamp_dt = current_timestamp_dt
                    latest_file = f
            except ValueError:
                logger.warning(f"Не удалось распарсить метку времени из имени файла: {f}. Игнорирую файл.")
                continue # Пропускаем файлы с некорректными метками времени
        else:
            logger.warning(f"Имя файла '{f}' не соответствует ожидаемому шаблону 'trade_model_YYYYMMDD_HHMM.pkl'. Игнорирую файл.")

    if latest_file is None:
        raise FileNotFoundError(f"Не найдено валидных файлов модели в папке '{model_dir}' по шаблону '{pattern}' с корректной меткой времени в имени.")

    logger.info(f"💾 Загружаю последнюю модель по метке времени в имени файла: {latest_file}")
    mdl = joblib.load(latest_file)
    return mdl["model"], mdl["features"]

# --- Глобальные переменные для модели и признаков ---
# Эти переменные будут инициализированы в initialize_bot
model = None
features = None

# --- НАСТРОЙКИ ЛОГИРОВАНИЯ ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler('bot_log.log', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# --- ФАЙЛЫ КОНФИГУРАЦИИ И СОСТОЯНИЯ ---
CONFIG_FILE = 'config.json'
STATE_FILE = 'bot_state.json'

# --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ИЗ КОНФИГА ---
SYMBOLS = []
TIMEFRAME = '5m'
MONITORING_INTERVAL_SECONDS = 60
MIN_SIGNAL_STRENGTH = 3 # Устанавливается из конфига
FILTER_THRESHOLD = 0.6 # Устанавливается из конфига
EXCHANGE = None
active_trades = {}  # Глобальный словарь для сделок

# --- НАСТРОЙКИ СТРАТЕГИИ ---
ATR_LENGTH = 14
MACD_FAST_LENGTH = 12
MACD_SLOW_LENGTH = 26
MACD_SIGNAL_LENGTH = 9
BB_LENGTH = 20
BB_MULTIPLIER = 2.0
BBW_EMA_LENGTH = 20
BBW_THRESHOLD_MULTIPLIER = 1.0
VOLUME_EMA_LENGTH = 20
VOLUME_CONFIRMATION_MULTIPLIER = 1.2
RSI_REVERSAL_LONG_THRESHOLD = 45 # поменять на 40
RSI_REVERSAL_SHORT_THRESHOLD = 55 # поменять на 60
MACD_REVERSAL_CONFIRMATION = True
BB_MIDDLE_CROSS_ATR_BUFFER = 0.2
BBW_EMA_LENGTH = 14 # Примерное значение, если не определено
VOLUME_EMA_LENGTH = 20
ADX_LENGTH = 14

# --- НОВЫЕ НАСТРОЙКИ ДЛЯ ИНДИКАТОРОВ ---
CMF_LENGTH = 20
RVI_LENGTH = 14
KAMA_LENGTH = 10
KAMA_FAST_EMA_PERIOD = 2
KAMA_SLOW_EMA_PERIOD = 30

# --- КОНСТАНТЫ ДЛЯ ДИНАМИЧЕСКИХ SL/TP ---
LOOKBACK_CANDLES_FOR_LEVELS = 20
LEVEL_PROXIMITY_ATR_MULTIPLIER = 0.5
MIN_PROFIT_ATR_MULTIPLIER_IF_NO_LEVELS = 1.5
MIN_SL_ATR_MULTIPLIER_IF_NO_LEVELS = 1.0
MIN_SL_PERCENTAGE_OF_ENTRY = 0.005
MIN_TP_PERCENTAGE_OF_ENTRY = 0.0075
MIN_RR_RATIO_TP1 = 1.0
MIN_SL_ATR_MULTIPLIER_FLOOR = 1.5
MIN_TP_STEP_PERCENTAGE = 0.005

# --- Имена колонок для индикаторов ---
MACD_SUFFIX = f'_{MACD_FAST_LENGTH}_{MACD_SLOW_LENGTH}_{MACD_SIGNAL_LENGTH}'
BB_SUFFIX = f'_{BB_LENGTH}_{BB_MULTIPLIER}'


def save_state(trades_to_save):
    """Сохраняет словарь активных сделок в JSON файл."""
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(trades_to_save, f, indent=4)
        logger.info(f"💾 Состояние успешно сохранено в {STATE_FILE}.")
    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении состояния в {STATE_FILE}: {e}", exc_info=True)


def load_state():
    """Загружает словарь активных сделок из JSON файла, проверяя целостность данных."""
    global active_trades
    if not os.path.exists(STATE_FILE):
        logger.info(f"Файл состояния {STATE_FILE} не найден. Начинаем новую сессию.")
        return

    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            loaded_trades = json.load(f)

        clean_trades = {}
        if loaded_trades and isinstance(loaded_trades, dict):
            for symbol, trade in loaded_trades.items():
                # Проверяем, что запись является словарем и содержит ключевые поля
                if isinstance(trade, dict) and 'type' in trade and 'entry_price' in trade:
                    clean_trades[symbol] = trade
                else:
                    logger.warning(f"   -> Обнаружена поврежденная запись для '{symbol}'. Запись проигнорирована. Данные: {trade}")
        
        active_trades = clean_trades
        if not active_trades:
            logger.info("Файл состояния не содержал валидных сделок. Начинаем новую сессию.")
        else:
            logger.info(f"✅ Состояние успешно загружено. Восстановлено валидных сделок: {len(active_trades)}")
            for symbol, trade in active_trades.items():
                logger.info(f"   -> Восстановлена сделка по {symbol}: {trade['type']} @ {trade['entry_price']}")

    except json.JSONDecodeError:
        logger.error(f"❌ Ошибка декодирования JSON из {STATE_FILE}. Файл поврежден. Начинаем новую сессию.")
        active_trades = {}
    except Exception as e:
        logger.error(f"❌ Не удалось загрузить состояние из {STATE_FILE}: {e}", exc_info=True)
        active_trades = {}


def load_config():
    """Загружает настройки бота из JSON файла конфигурации."""
    global SYMBOLS, TIMEFRAME, MONITORING_INTERVAL_SECONDS, MIN_SIGNAL_STRENGTH, FILTER_THRESHOLD

    if not os.path.exists(CONFIG_FILE):
        logger.error(f"Файл конфигурации '{CONFIG_FILE}' не найден. Создайте его с необходимыми настройками.")
        default_config = {
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "timeframe": "5m",
            "monitoring_interval_seconds": 60,
            "min_signal_strength": 3,
            "filter_threshold": 0.6
        }
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4)
        logger.info(f"Создан дефолтный файл конфигурации '{CONFIG_FILE}'. Пожалуйста, отредактируйте его.")
        exit()

    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        SYMBOLS = config.get('symbols', [])
        TIMEFRAME = config.get('timeframe', '5m')
        MONITORING_INTERVAL_SECONDS = config.get('monitoring_interval_seconds', 60)
        MIN_SIGNAL_STRENGTH = config.get('min_signal_strength', 3)
        FILTER_THRESHOLD = config.get('filter_threshold', 0.6) # Загружаем порог ML-фильтра

        if not SYMBOLS:
            logger.error(f"Список символов в '{CONFIG_FILE}' пуст. Пожалуйста, укажите символы для анализа.")
            exit()

        logger.info(f"Настройки бота успешно загружены. Сила сигнала для входа: {MIN_SIGNAL_STRENGTH}, Порог ML-фильтра: {FILTER_THRESHOLD}")
    except json.JSONDecodeError as e:
        logger.critical(f"Ошибка декодирования JSON из файла конфигурации '{CONFIG_FILE}': {e}", exc_info=True)
        exit()
    except Exception as e:
        logger.critical(f"Неизвестная ошибка при загрузке конфигурации из '{CONFIG_FILE}': {e}", exc_info=True)
        exit()


def retry_on_exception(func, retries=3, delay=1, backoff=2):
    """Выполняет функцию с повторными попытками при возникновении определенных исключений."""
    for i in range(retries):
        try:
            return func()
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout, ccxt.DDoSProtection, ccxt.ExchangeError) as e:
            logger.warning(f"Ошибка биржи (попытка {i + 1}/{retries}): {e}. Повтор через {delay:.1f} сек...")
            time.sleep(delay)
            delay *= backoff
        except Exception as e:
            logger.error(f"Неизвестная ошибка (попытка {i + 1}/{retries}): {e}. Повтор через {delay:.1f} сек...", exc_info=True)
            time.sleep(delay)
            delay *= backoff
    raise Exception(f"Все {retries} попыток исчерпаны. Не удалось выполнить функцию {func.__name__}.")


def fetch_data(symbol):
    """Загружает исторические данные (свечи) для конкретной пары с биржи."""
    def _fetch():
        ohlcv = EXCHANGE.fetch_ohlcv(symbol, TIMEFRAME, limit=5000)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df

    try:
        return retry_on_exception(_fetch, retries=5, delay=2)
    except Exception as e:
        logger.critical(f"[{symbol}] Критическая ошибка: Не удалось получить данные: {e}", exc_info=True)
        return None


def calculate_atr_manually(df, length=14):
    """Расчет Average True Range (ATR) вручную."""
    if len(df) < length + 1:
        df[f'ATR_{length}'] = np.nan
        return df
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close}).max(axis=1)
    df[f'ATR_{length}'] = tr.ewm(span=length, adjust=False).mean()
    return df


def add_indicators(df, symbol):
    """Добавляет технические индикаторы в DataFrame."""
    if df.empty:
        print(f"[{symbol}] Входной DataFrame пуст. Возвращаю пустой DF.")
        return df

    df_copy = df.copy()
    
    # 1. Убедимся, что колонка 'timestamp' существует и является datetime
    if 'timestamp' not in df_copy.columns:
        print(f"[{symbol}] Ошибка: Колонка 'timestamp' отсутствует в DataFrame. Не могу установить DatetimeIndex.")
        return pd.DataFrame()
    
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], unit='ms', errors='coerce')
    df_copy.dropna(subset=['timestamp'], inplace=True)
    if df_copy.empty:
        print(f"[{symbol}] WARNING: DataFrame пуст после очистки некорректных временных меток.")
        return pd.DataFrame()

    # 2. Приводим все числовые колонки к float и удаляем бесконечности
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            # Исправлено: заменено inplace=True на прямое присваивание
            df_copy[col] = df_copy[col].replace([np.inf, -np.inf], np.nan)
            
    # Дополнительная очистка NaNs в основных колонках, необходимых для расчетов
    df_copy.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
    if df_copy.empty:
        print(f"[{symbol}] WARNING: DataFrame пуст после очистки основных OHLCV данных.")
        return pd.DataFrame()

    # 3. Установим 'timestamp' как индекс DataFrame и отсортируем
    df_copy.set_index('timestamp', inplace=True)
    df_copy.sort_index(inplace=True) 

    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df_copy.columns for col in required_cols):
        print(f"[{symbol}] Ошибка: Отсутствуют необходимые колонки ({', '.join(required_cols)}) для расчета индикаторов.")
        df_copy.reset_index(inplace=True, drop=False)
        return pd.DataFrame()

    # --- РАСЧЕТ ИНДИКАТОРОВ ---

    # ATR (рассчитывается вручную)
    df_copy = calculate_atr_manually(df_copy, length=ATR_LENGTH)

    # EMA
    df_copy.ta.ema(length=50, append=True, col_names=(f'EMA_50',))
    df_copy.ta.ema(length=200, append=True, col_names=(f'EMA_200',))
    # RSI
    df_copy.ta.rsi(length=14, append=True, col_names=(f'RSI_14',))
    # MACD
    df_copy.ta.macd(fast=MACD_FAST_LENGTH, slow=MACD_SLOW_LENGTH, signal=MACD_SIGNAL_LENGTH, append=True)
    # Bollinger Bands
    df_copy.ta.bbands(length=BB_LENGTH, std=BB_MULTIPLIER, append=True)
    
    # Переименовываем колонки BBands
    bb_upper_col = f'BBU_{BB_LENGTH}_{BB_MULTIPLIER}'
    bb_middle_col = f'BBM_{BB_LENGTH}_{BB_MULTIPLIER}'
    bb_lower_col = f'BBL_{BB_LENGTH}_{BB_MULTIPLIER}'
    
    if all(col in df_copy.columns for col in [bb_upper_col, bb_middle_col, bb_lower_col]):
        df_copy.rename(columns={
            bb_upper_col: 'BB_upper',
            bb_middle_col: 'BB_middle',
            bb_lower_col: 'BB_lower'
        }, inplace=True)
    else:
        print(f"[{symbol}] WARNING: Не все колонки BBands найдены для переименования.")
        for col in ['BB_upper', 'BB_middle', 'BB_lower']:
            if col not in df_copy.columns:
                df_copy[col] = np.nan

    # Расчет BB_width
    df_copy['BB_width'] = (df_copy['BB_upper'] - df_copy['BB_lower']) / df_copy['BB_middle'].replace(0, np.nan)
    df_copy['BB_width'] = df_copy['BB_width'].replace([np.inf, -np.inf], np.nan)

    # BBW EMA
    bbw_col_name = 'BB_width' 
    bbw_ema_col_name = f'BBW_EMA_{BBW_EMA_LENGTH}'
    if bbw_col_name in df_copy.columns and not df_copy[bbw_col_name].isnull().all():
        df_copy[bbw_ema_col_name] = df_copy[bbw_col_name].ewm(span=BBW_EMA_LENGTH, adjust=False).mean()
    else:
        print(f"[{symbol}] WARNING: 'BB_width' не содержит данных для расчета BBW_EMA. Колонка '{bbw_ema_col_name}' будет NaN.")
        df_copy[bbw_ema_col_name] = np.nan # Убедимся, что колонка существует

    # Volume EMA
    volume_ema_col_name = 'VOLUME_EMA' 
    if 'volume' in df_copy.columns and not df_copy['volume'].isnull().all():
        df_copy[volume_ema_col_name] = df_copy['volume'].ewm(span=VOLUME_EMA_LENGTH, adjust=False).mean()
    else:
        print(f"[{symbol}] WARNING: 'volume' колонка отсутствует или пуста. Не могу рассчитать VOLUME_EMA.")
        df_copy[volume_ema_col_name] = np.nan # Убедимся, что колонка существует
    
    # Chaikin Money Flow (CMF)
    if 'volume' in df_copy.columns and not (df_copy['volume'].isnull().all() or (df_copy['volume'] == 0).all()):
        df_copy.ta.cmf(length=CMF_LENGTH, append=True, col_names=(f'CMF_{CMF_LENGTH}',))
    else:
        print(f"[{symbol}] WARNING: Объем отсутствует или равен нулю, CMF не будет рассчитан.")
        
    cmf_col = f'CMF_{CMF_LENGTH}'
    if cmf_col not in df_copy.columns:
        df_copy[cmf_col] = np.nan
        print(f"[{symbol}] WARNING: Колонка '{cmf_col}' не была создана pandas_ta. Добавлена с NaN.")
    
    # Relative Volatility Index (RVI)
    df_copy.ta.rvi(length=RVI_LENGTH, append=True, col_names=(f'RVI_{RVI_LENGTH}',))
    rvi_col = f'RVI_{RVI_LENGTH}'
    if rvi_col not in df_copy.columns:
        df_copy[rvi_col] = np.nan
        print(f"[{symbol}] WARNING: Колонка '{rvi_col}' не была создана pandas_ta. Добавлена с NaN.")
    
    # Kaufman's Adaptive Moving Average (KAMA)
    df_copy.ta.kama(length=KAMA_LENGTH, fast=KAMA_FAST_EMA_PERIOD, slow=KAMA_SLOW_EMA_PERIOD, append=True, col_names=(f'KAMA_{KAMA_LENGTH}_{KAMA_FAST_EMA_PERIOD}_{KAMA_SLOW_EMA_PERIOD}',))
    kama_col = f'KAMA_{KAMA_LENGTH}_{KAMA_FAST_EMA_PERIOD}_{KAMA_SLOW_EMA_PERIOD}'
    if kama_col not in df_copy.columns:
        df_copy[kama_col] = np.nan
        print(f"[{symbol}] WARNING: Колонка '{kama_col}' не была создана pandas_ta. Добавлена с NaN.")

    # # Stochastic Oscillator (STOCH) - УДАЛЕНО
    # # Проверяем, есть ли вариации в high/low, иначе Stoch будет NaN
    # if (df_copy['high'] == df_copy['low']).all():
    #     print(f"[{symbol}] WARNING: Цены не меняются (High == Low). Стохастик не будет рассчитан.")
    #     df_copy['STOCH_k'] = np.nan
    #     df_copy['STOCH_d'] = np.nan
    # else:
    #     df_copy.ta.stoch(k=STOCH_K_LENGTH, d=STOCH_D_LENGTH, append=True)
    #     stoch_k_col_name = f'STOCHk_{STOCH_K_LENGTH}_{STOCH_D_LENGTH}'
    #     stoch_d_col_name = f'STOCHd_{STOCH_K_LENGTH}_{STOCH_D_LENGTH}'
    #     if stoch_k_col_name in df_copy.columns and stoch_d_col_name in df_copy.columns:
    #         df_copy.rename(columns={stoch_k_col_name: 'STOCH_k', stoch_d_col_name: 'STOCH_d'}, inplace=True)
    #     else:
    #         print(f"[{symbol}] WARNING: Стохастик ({stoch_k_col_name}, {stoch_d_col_name}) не был рассчитан. Возможно, недостаточно данных.")
    #         df_copy['STOCH_k'] = np.nan
    #         df_copy['STOCH_d'] = np.nan
    
    # Average Directional Index (ADX)
    df_copy.ta.adx(length=ADX_LENGTH, append=True)
    adx_col_name = f'ADX_{ADX_LENGTH}'
    if adx_col_name in df_copy.columns:
        df_copy.rename(columns={adx_col_name: 'ADX_14'}, inplace=True)
    else:
        print(f"[{symbol}] WARNING: ADX ({adx_col_name}) не был рассчитан. Возможно, недостаточно данных.")
        df_copy['ADX_14'] = np.nan

    # # Commodity Channel Index (CCI) - УДАЛЕНО
    # # CCI также чувствителен к "плоским" свечам
    # if (df_copy['high'] == df_copy['low']).all():
    #     print(f"[{symbol}] WARNING: Цены не меняются (High == Low). CCI не будет рассчитан.")
    #     df_copy['CCI_20'] = np.nan
    # else:
    #     df_copy.ta.cci(length=CCI_LENGTH, append=True)
    #     cci_col_name = f'CCI_{CCI_LENGTH}'
    #     if cci_col_name in df_copy.columns:
    #         df_copy.rename(columns={cci_col_name: 'CCI_20'}, inplace=True)
    #     else:
    #         print(f"[{symbol}] WARNING: CCI ({cci_col_name}) не был рассчитан. Возможно, недостаточно данных.")
    #         df_copy['CCI_20'] = np.nan

    # # Volume Weighted Average Price (VWAP) - УДАЛЕНО
    # # VWAP критически зависит от объема.
    # if 'volume' in df_copy.columns and not (df_copy['volume'].isnull().all() or (df_copy['volume'] == 0).all()):
    #     try:
    #         df_copy.ta.vwap(append=True)
    #         df_copy.rename(columns={'VWAP': 'VWAP_14'}, inplace=True) # pandas_ta.vwap() не использует length
    #     except Exception as e:
    #         print(f"[{symbol}] WARNING: Ошибка при расчете VWAP: {e}. Возможно, недостаточно данных или проблемы с индексом/данными.")
    #         df_copy['VWAP_14'] = np.nan
    # else:
    #     print(f"[{symbol}] WARNING: Объем отсутствует или равен нулю, VWAP не будет рассчитан.")
    #     if 'VWAP_14' not in df_copy.columns: # Убедимся, что колонка существует
    #         df_copy['VWAP_14'] = np.nan

    # --- ЗАПОЛНЕНИЕ NaN для всех индикаторов ---
    # Исправлено: заменено inplace=True на прямое присваивание
    for col in df_copy.columns:
        if df_copy[col].dtype == 'float64' and df_copy[col].isnull().any():
            df_copy[col] = df_copy[col].fillna(0)

    # --- ЗАВЕРШАЮЩИЕ ШАГИ ---
    # Вернуть 'timestamp' как обычную колонку
    df_copy.reset_index(inplace=True, drop=False) 

    # Финальная проверка на NaN.
    initial_rows_before_final_dropna = len(df_copy)
    df_copy.dropna(inplace=True) 
    final_rows_after_final_dropna = len(df_copy)

    if final_rows_after_final_dropna == 0:
        print(f"[{symbol}] CRITICAL: DataFrame пуст после финальной очистки, несмотря на попытки заполнения NaN. Это серьезная проблема с данными.")
        return pd.DataFrame()
    elif final_rows_after_final_dropna < initial_rows_before_final_dropna:
        print(f"[{symbol}] WARNING: После финальной очистки удалено {initial_rows_before_final_dropna - final_rows_after_final_dropna} строк. Осталось {final_rows_after_final_dropna} строк.")
    #else:
        #print(f"[{symbol}] INFO: DataFrame успешно обработан, содержит {final_rows_after_final_dropna} строк.")
    
    return df_copy


def find_significant_levels(df, current_price, position_type, current_atr):
    """Ищет потенциальные уровни поддержки/сопротивления."""
    levels = []
    if len(df) > LOOKBACK_CANDLES_FOR_LEVELS:
        recent_candles = df.iloc[-(LOOKBACK_CANDLES_FOR_LEVELS + 1):-1]
        if position_type == "LONG":
            levels.extend(recent_candles['high'].tolist())
        elif position_type == "SHORT":
            levels.extend(recent_candles['low'].tolist())
    last_candle = df.iloc[-1]
    if not pd.isna(last_candle['BB_upper']): # Используем переименованные колонки
        levels.append(last_candle['BB_upper'])
    if not pd.isna(last_candle['BB_lower']): # Используем переименованные колонки
        levels.append(last_candle['BB_lower'])
    if not pd.isna(last_candle['EMA_50']):
        levels.append(last_candle['EMA_50'])
    if not pd.isna(last_candle['EMA_200']):
        levels.append(last_candle['EMA_200'])
    levels = list(set([l for l in levels if pd.notna(l) and np.isfinite(l)]))
    if position_type == "LONG":
        tp_levels = sorted([l for l in levels if l > current_price])
        sl_levels = sorted([l for l in levels if l < current_price], reverse=True)
    else:  # SHORT
        tp_levels = sorted([l for l in levels if l < current_price], reverse=True)
        sl_levels = sorted([l for l in levels if l > current_price])
    
    filtered_tp_levels = [l for l in tp_levels if (position_type == "LONG" and l > current_price + current_atr * LEVEL_PROXIMITY_ATR_MULTIPLIER) or (position_type == "SHORT" and l < current_price - current_atr * LEVEL_PROXIMITY_ATR_MULTIPLIER)][:3]
    filtered_sl_levels = [l for l in sl_levels if (position_type == "LONG" and l < current_price - current_atr * LEVEL_PROXIMITY_ATR_MULTIPLIER) or (position_type == "SHORT" and l > current_price + current_atr * LEVEL_PROXIMITY_ATR_MULTIPLIER)][:1]
    
    return filtered_sl_levels, filtered_tp_levels
def calculate_dynamic_sl_tp(entry_price, df, position_type, symbol, signal_type="GENERIC"):
    """
    Рассчитывает динамический стоп-лосс и тейк-профиты с гарантированным минимальным расстоянием 
    и правильной последовательностью TP, используя процентный шаг для дешевых активов.
    """
    last_candle = df.iloc[-1]
    current_atr = last_candle.get(f'ATR_{ATR_LENGTH}', np.nan)
    if pd.isna(current_atr) or not np.isfinite(current_atr) or current_atr <= 1e-10:
        current_atr = entry_price * 0.001
        logger.warning(f"[{symbol}] ATR невалиден, используется % от цены для расчета.")

    sl_price = 0
    tp_prices = {}
    potential_sl_levels, potential_tp_levels = find_significant_levels(df, entry_price, position_type, current_atr)

    # --- 1. Расчет SL по основной логике ---
    if position_type == "LONG":
        if "HAMMER" in signal_type and last_candle['low'] < entry_price:
            sl_price = last_candle['low']
        elif "ENGULFING" in signal_type and df.iloc[-2]['low'] < entry_price:
            sl_price = df.iloc[-2]['low']
        elif potential_sl_levels:
            sl_candidate = potential_sl_levels[0]
            sl_price = sl_candidate if sl_candidate < entry_price - (current_atr * 0.1) else entry_price - (current_atr * MIN_SL_ATR_MULTIPLIER_IF_NO_LEVELS)
        else:
            sl_price = entry_price - (current_atr * MIN_SL_ATR_MULTIPLIER_IF_NO_LEVELS)
    elif position_type == "SHORT":
        if "ENGULFING" in signal_type and df.iloc[-2]['high'] > entry_price:
            sl_price = df.iloc[-2]['high']
        elif potential_sl_levels:
            sl_candidate = potential_sl_levels[0]
            sl_price = sl_candidate if sl_candidate > entry_price + (current_atr * 0.1) else entry_price + (current_atr * MIN_SL_ATR_MULTIPLIER_IF_NO_LEVELS)
        else:
            sl_price = entry_price + (current_atr * MIN_SL_ATR_MULTIPLIER_IF_NO_LEVELS)

    # --- 2. Гарантируем минимальное расстояние для SL ---
    min_sl_dist_percent = entry_price * MIN_SL_PERCENTAGE_OF_ENTRY
    min_sl_dist_atr = current_atr * MIN_SL_ATR_MULTIPLIER_FLOOR
    final_min_dist = max(min_sl_dist_percent, min_sl_dist_atr)

    if position_type == "LONG":
        guaranteed_sl = entry_price - final_min_dist
        sl_price = min(sl_price, guaranteed_sl)
    elif position_type == "SHORT":
        guaranteed_sl = entry_price + final_min_dist
        sl_price = max(sl_price, guaranteed_sl)

    # --- 3. Первичное назначение TP ---
    if potential_tp_levels:
        tp_prices['TP1'] = potential_tp_levels[0] if len(potential_tp_levels) > 0 else None
        tp_prices['TP2'] = potential_tp_levels[1] if len(potential_tp_levels) > 1 else None
        tp_prices['TP3'] = potential_tp_levels[2] if len(potential_tp_levels) > 2 else None
    
    # --- 4. Корректировка TP1 для R:R ---
    risk_distance = abs(entry_price - sl_price)
    reward_distance_tp1 = abs(tp_prices.get('TP1', entry_price) - entry_price) if tp_prices.get('TP1') else 0
    if risk_distance > 1e-10 and reward_distance_tp1 < risk_distance * MIN_RR_RATIO_TP1:
        logger.warning(f"[{symbol}] TP1 скорректирован для соблюдения R:R > {MIN_RR_RATIO_TP1}")
        if position_type == "LONG":
            tp_prices['TP1'] = entry_price + (risk_distance * MIN_RR_RATIO_TP1)
        else:
            tp_prices['TP1'] = entry_price - (risk_distance * MIN_RR_RATIO_TP1)
            
    # --- 5. Каскадный пересчет TP с использованием процентного шага ---
    atr_based_step = current_atr * (MIN_PROFIT_ATR_MULTIPLIER_IF_NO_LEVELS / 2)
    percent_based_step = entry_price * MIN_TP_STEP_PERCENTAGE
    
    final_tp_step = max(atr_based_step, percent_based_step)
    logger.info(f"[{symbol}] Шаг для TP определен как {final_tp_step:.8f} (ATR_step: {atr_based_step:.8f}, Percent_step: {percent_based_step:.8f})")

    if position_type == "LONG":
        if not tp_prices.get('TP1'):
            min_tp1_price = entry_price + (entry_price * MIN_TP_PERCENTAGE_OF_ENTRY)
            tp_prices['TP1'] = max(min_tp1_price, entry_price + (current_atr * MIN_PROFIT_ATR_MULTIPLIER_IF_NO_LEVELS))
        
        if not tp_prices.get('TP2') or tp_prices.get('TP2') <= tp_prices.get('TP1'):
            tp_prices['TP2'] = tp_prices.get('TP1') + final_tp_step
        
        if not tp_prices.get('TP3') or tp_prices.get('TP3') <= tp_prices.get('TP2'):
            tp_prices['TP3'] = tp_prices.get('TP2') + final_tp_step
    
    elif position_type == "SHORT":
        if not tp_prices.get('TP1'):
            min_tp1_price = entry_price - (entry_price * MIN_TP_PERCENTAGE_OF_ENTRY)
            tp_prices['TP1'] = min(min_tp1_price, entry_price - (current_atr * MIN_PROFIT_ATR_MULTIPLIER_IF_NO_LEVELS))

        if not tp_prices.get('TP2') or tp_prices.get('TP2') >= tp_prices.get('TP1'):
            tp_prices['TP2'] = tp_prices.get('TP1') - final_tp_step
        
        if not tp_prices.get('TP3') or tp_prices.get('TP3') >= tp_prices.get('TP2'):
            tp_prices['TP3'] = tp_prices.get('TP2') - final_tp_step

    return sl_price, tp_prices


def check_hammer_candlestick(df):
    """Проверяет наличие паттерна Молот (Hammer) на последней свече."""
    if len(df) < 1: return False
    last_candle = df.iloc[-1]
    body = abs(last_candle['open'] - last_candle['close'])
    lower_shadow = min(last_candle['open'], last_candle['close']) - last_candle['low']
    upper_shadow = last_candle['high'] - max(last_candle['open'], last_candle['close'])
    candle_range = last_candle['high'] - last_candle['low']
    if candle_range < 1e-9: return False

    is_small_body = body < candle_range * 0.3
    is_long_lower_shadow = lower_shadow >= 2 * body if body > 1e-9 else True
    is_small_upper_shadow = upper_shadow <= body * 0.3 if body > 1e-9 else True
    return is_small_body and is_long_lower_shadow and is_small_upper_shadow


def check_engulfing_candlestick(df, is_bullish):
    """Проверяет наличие паттерна Поглощение (Engulfing) на последних двух свечах."""
    if len(df) < 2: return False
    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    if is_bullish:
        is_prev_bearish = prev_candle['close'] < prev_candle['open']
        is_curr_bullish = last_candle['close'] > last_candle['open']
        is_engulfing = (last_candle['open'] <= prev_candle['close'] and last_candle['close'] >= prev_candle['open'])
        return is_prev_bearish and is_curr_bullish and is_engulfing
    else:  # is_bearish
        is_prev_bullish = prev_candle['close'] > prev_candle['open']
        is_curr_bearish = last_candle['close'] < last_candle['open']
        is_engulfing = (last_candle['open'] >= prev_candle['close'] and last_candle['close'] <= prev_candle['open'])
        return is_prev_bullish and is_curr_bearish and is_engulfing


def is_bollinger_bands_squeezing(df, current_bbw, bbw_ema):
    """Проверяет, находятся ли полосы Боллинджера в состоянии сжатия."""
    return current_bbw < bbw_ema * BBW_THRESHOLD_MULTIPLIER

# --- Функция ML-фильтра ---
def filter_signal(candle_dict: dict) -> float:
    # Убедитесь, что 'features' и 'model' доступны глобально
    global features, model

    if model is None or features is None:
        logger.error("ML модель или список признаков не загружены.")
        return 0.0 # Возвращаем 0, если модель не готова

    feat_vector = []
    for f in features:
        if f not in candle_dict:
            logger.warning(f"Отсутствует признак '{f}' для ML-модели. Возвращаю 0.00.")
            return 0.0 # Если признак отсутствует, возвращаем 0
        feat_vector.append(candle_dict[f])

    # Используем predict_proba для получения вероятности класса 1 (прибыльной сделки)
    prob = model.predict_proba([feat_vector])[:, 1][0]
    return prob


def analyze_data(symbol, df):
    """Анализирует данные и, если сигнал достаточно сильный, открывает сделку."""
    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    current_close_price = last_candle['close']

    long_score, short_score = 0, 0
    long_signal_reasons, short_signal_reasons = [], []
    signal_type_long, signal_type_short = "GENERIC_LONG", "GENERIC_SHORT" # Изменено для большей ясности

    # --- Существующие индикаторы ---
    if (prev_candle[f'MACD{MACD_SUFFIX}'] < prev_candle[f'MACDs{MACD_SUFFIX}'] and last_candle[f'MACD{MACD_SUFFIX}'] > last_candle[f'MACDs{MACD_SUFFIX}']):
        long_score += 1
        long_signal_reasons.append("MACD кроссовер вверх")
    if (prev_candle[f'MACD{MACD_SUFFIX}'] > prev_candle[f'MACDs{MACD_SUFFIX}'] and last_candle[f'MACD{MACD_SUFFIX}'] < last_candle[f'MACDs{MACD_SUFFIX}']):
        short_score += 1
        short_signal_reasons.append("MACD кроссовер вниз")
    
    if (prev_candle['RSI_14'] < 30 and last_candle['RSI_14'] > 30):
        long_score += 1
        long_signal_reasons.append(f"RSI выход из перепроданности ({last_candle['RSI_14']:.2f})")
    if (prev_candle['RSI_14'] > 70 and last_candle['RSI_14'] < 70):
        short_score += 1
        short_signal_reasons.append(f"RSI выход из перекупленности ({last_candle['RSI_14']:.2f})")
    
    # Использование KAMA вместо EMA 50/200 для подтверждения тренда
    kama_col_name = f'KAMA_{KAMA_LENGTH}_{KAMA_FAST_EMA_PERIOD}_{KAMA_SLOW_EMA_PERIOD}'
    if kama_col_name in last_candle:
        # Проверка на восходящий тренд по KAMA
        if current_close_price > last_candle[kama_col_name] and last_candle[kama_col_name] > df.iloc[-2][kama_col_name]: # Цена выше KAMA и KAMA растет
            long_score += 1
            long_signal_reasons.append("Восходящий тренд по KAMA")
        # Проверка на нисходящий тренд по KAMA
        if current_close_price < last_candle[kama_col_name] and last_candle[kama_col_name] < df.iloc[-2][kama_col_name]: # Цена ниже KAMA и KAMA падает
            short_score += 1
            short_signal_reasons.append("Нисходящий тренд по KAMA")
    else: # Если KAMA по каким-то причинам не рассчитана, используем старые EMA
        if (current_close_price > last_candle['EMA_50'] and last_candle['EMA_50'] > last_candle['EMA_200']):
            long_score += 1
            long_signal_reasons.append("Сильный восходящий тренд (EMA)")
        if (current_close_price < last_candle['EMA_50'] and last_candle['EMA_50'] < last_candle['EMA_200']):
            short_score += 1
            short_signal_reasons.append("Сильный нисходящий тренд (EMA)")

    bbw_col_name = 'BB_width' # Теперь используем переименованную колонку
    bbw_ema_col_name = f'BBW_EMA_{BBW_EMA_LENGTH}'
    if bbw_col_name in last_candle and bbw_ema_col_name in last_candle and is_bollinger_bands_squeezing(df, last_candle[bbw_col_name], last_candle[bbw_ema_col_name]):
        long_score += 1
        short_score += 1
        long_signal_reasons.append("BB Squeeze")
        short_signal_reasons.append("BB Squeeze")
    
    # VOLUME_EMA - теперь без суффикса, как в feature_engineering.py
    if last_candle['volume'] > (last_candle['VOLUME_EMA'] * VOLUME_CONFIRMATION_MULTIPLIER):
        long_score += 1
        short_score += 1
        long_signal_reasons.append("Повышенный объем")
        short_signal_reasons.append("Повышенный объем")

    # --- НОВЫЕ ИНДИКАТОРЫ И ЛОГИКА ---
    # Chaikin Money Flow (CMF)
    cmf_col_name = f'CMF_{CMF_LENGTH}'
    if cmf_col_name in last_candle:
        # Для лонга: CMF > 0 и растет или CMF пересек 0 снизу вверх
        if last_candle[cmf_col_name] > 0.05 and prev_candle[cmf_col_name] <= 0.05: # Пересечение нуля снизу вверх или уверенное положительное значение
            long_score += 1
            long_signal_reasons.append(f"CMF (покупка) ({last_candle[cmf_col_name]:.2f})")
        elif last_candle[cmf_col_name] > 0.1 and last_candle[cmf_col_name] > prev_candle[cmf_col_name]: # Уверенное положительное значение и рост
            long_score += 0.5
            long_signal_reasons.append(f"CMF (сильное давление покупки) ({last_candle[cmf_col_name]:.2f})")

        # Для шорта: CMF < 0 и падает или CMF пересек 0 сверху вниз
        if last_candle[cmf_col_name] < -0.05 and prev_candle[cmf_col_name] >= -0.05: # Пересечение нуля сверху вниз или уверенное отрицательное значение
            short_score += 1
            short_signal_reasons.append(f"CMF (продажа) ({last_candle[cmf_col_name]:.2f})")
        elif last_candle[cmf_col_name] < -0.1 and last_candle[cmf_col_name] < prev_candle[cmf_col_name]: # Уверенное отрицательное значение и падение
            short_score += 0.5
            short_signal_reasons.append(f"CMF (сильное давление продажи) ({last_candle[cmf_col_name]:.2f})")
    
    # --- Свечные паттерны ---
    if current_close_price < last_candle['EMA_50']: # Паттерны для лонга чаще ищут на коррекции/падении
        if check_hammer_candlestick(df):
            long_score += 2
            long_signal_reasons.append("Паттерн: Молот")
            signal_type_long += "_HAMMER"
        if check_engulfing_candlestick(df, is_bullish=True):
            long_score += 2
            long_signal_reasons.append("Паттерн: Бычье поглощение")
            signal_type_long += "_ENGULFING"
    
    if current_close_price > last_candle['EMA_50']: # Паттерны для шорта чаще ищут на росте/вершине
        if check_engulfing_candlestick(df, is_bullish=False):
            short_score += 2
            short_signal_reasons.append("Паттерн: Медвежье поглощение")
            signal_type_short += "_ENGULFING"
            
    # --- Открытие сделки ---
    if long_score >= MIN_SIGNAL_STRENGTH:
        # ▶ ML-фильтр
        prob = filter_signal(last_candle.to_dict())
        if prob < FILTER_THRESHOLD:
            logger.info(f"[{symbol}] ML-фильтр отклонил LONG (prob={prob:.2f} < {FILTER_THRESHOLD})")
            return
        
        sl, tps = calculate_dynamic_sl_tp(current_close_price, df, "LONG", symbol, signal_type_long)
        active_trades[symbol] = {"type": "LONG", "entry_price": current_close_price, "sl": sl, "tp1": tps.get('TP1'), "tp2": tps.get('TP2'), "tp3": tps.get('TP3'), "status": "active"}
        logger.info(f"✅ [{symbol}] ОТКРЫТА LONG СДЕЛКА (Сила: {long_score}, ML-Prob: {prob:.2f}) @ {current_close_price:.8f}")
        logger.info(f"   SL: {sl:.8f}, TP1: {tps.get('TP1', 0.0):.8f}, TP2: {tps.get('TP2', 0.0):.8f}, TP3: {tps.get('TP3', 0.0):.8f}")
        logger.info(f"   Причины: {'; '.join(long_signal_reasons)}")
        save_state(active_trades)

    elif short_score >= MIN_SIGNAL_STRENGTH:
        # ▶ ML-фильтр
        prob = filter_signal(last_candle.to_dict())
        if prob < FILTER_THRESHOLD:
            logger.info(f"[{symbol}] ML-фильтр отклонил SHORT (prob={prob:.2f} < {FILTER_THRESHOLD})")
            return

        sl, tps = calculate_dynamic_sl_tp(current_close_price, df, "SHORT", symbol, signal_type_short)
        active_trades[symbol] = {"type": "SHORT", "entry_price": current_close_price, "sl": sl, "tp1": tps.get('TP1'), "tp2": tps.get('TP2'), "tp3": tps.get('TP3'), "status": "active"}
        logger.info(f"✅ [{symbol}] ОТКРЫТА SHORT СДЕЛКА (Сила: {short_score}, ML-Prob: {prob:.2f}) @ {current_close_price:.8f}")
        logger.info(f"   SL: {sl:.8f}, TP1: {tps.get('TP1', 0.0):.8f}, TP2: {tps.get('TP2', 0.0):.8f}, TP3: {tps.get('TP3', 0.0):.8f}")
        logger.info(f"   Причины: {'; '.join(short_signal_reasons)}")
        save_state(active_trades)


def monitor_trade(symbol, df):
    """Отслеживает активную сделку, проверяя достижение SL/TP."""
    global active_trades
    trade = active_trades[symbol]
    current_price = df.iloc[-1]['close']
    trade_type = trade['type']
    status = trade['status']
    sl, tp1, tp2, tp3 = trade['sl'], trade['tp1'], trade['tp2'], trade['tp3']

    logger.info(f"👀 [{symbol}] Отслеживаю {trade_type} | Цена: {current_price:.8f} | Вход: {trade['entry_price']:.8f} | Статус: {status}")

    if trade_type == 'LONG':
        if current_price <= sl:
            logger.info(f"🛑 [{symbol}] СДЕЛКА ЗАКРЫТА по Stop Loss @ {current_price:.8f}")
            del active_trades[symbol]
            save_state(active_trades)
            return
        if tp3 and current_price >= tp3:
            logger.info(f"🎉 [{symbol}] СДЕЛКА ЗАКРЫТА по Take Profit 3 @ {current_price:.8f}")
            del active_trades[symbol]
            save_state(active_trades)
            return
        if tp2 and current_price >= tp2 and status != 'tp2_hit':
            logger.info(f"🎯 [{symbol}] Take Profit 2 достигнут @ {current_price:.8f}")
            trade['status'] = 'tp2_hit'
            save_state(active_trades)
        elif tp1 and current_price >= tp1 and status == 'active':
            logger.info(f"🎯 [{symbol}] Take Profit 1 достигнут @ {current_price:.8f}")
            trade['status'] = 'tp1_hit'
            save_state(active_trades)

    elif trade_type == 'SHORT':
        if current_price >= sl:
            logger.info(f"🛑 [{symbol}] СДЕЛКА ЗАКРЫТА по Stop Loss @ {current_price:.8f}")
            del active_trades[symbol]
            save_state(active_trades)
            return
        if tp3 and current_price <= tp3:
            logger.info(f"🎉 [{symbol}] СДЕЛКА ЗАКРЫТА по Take Profit 3 @ {current_price:.8f}")
            del active_trades[symbol]
            save_state(active_trades)
            return
        if tp2 and current_price <= tp2 and status != 'tp2_hit':
            logger.info(f"🎯 [{symbol}] Take Profit 2 достигнут @ {current_price:.8f}")
            trade['status'] = 'tp2_hit'
            save_state(active_trades)
        elif tp1 and current_price <= tp1 and status == 'active':
            logger.info(f"🎯 [{symbol}] Take Profit 1 достигнут @ {current_price:.8f}")
            trade['status'] = 'tp1_hit'
            save_state(active_trades)


def initialize_bot():
    """Инициализирует бота: загружает конфиг, подключается к бирже, загружает состояние и модель."""
    global EXCHANGE, model, features
    load_config()
    try:
        EXCHANGE = ccxt.binanceusdm({'options': {'defaultType': 'future', 'adjustForTimeDifference': True}})
        EXCHANGE.load_markets()
        logger.info("Биржа Binance USDT-M Futures успешно инициализирована.")
    except Exception as e:
        logger.critical(f"Ошибка при инициализации биржи: {e}", exc_info=True)
        exit()
    load_state()
    try:
        model, features = load_latest_model()
        logger.info("✅ ML-модель успешно загружена.")
    except Exception as e:
        logger.critical(f"❌ Ошибка при загрузке ML-модели: {e}. Бот не будет использовать ML-фильтр.", exc_info=True)
        model = None
        features = None


def main_loop():
    """Основной цикл работы бота."""
    initialize_bot()
    while True:
        logger.info(f"\n--- Новая итерация | Активных сделок: {len(active_trades)} ---")
        for symbol in SYMBOLS:
            df = fetch_data(symbol)
            if df is None or df.empty:
                continue
            df_with_indicators = add_indicators(df, symbol)
            if df_with_indicators.empty:
                continue
            try:
                if symbol in active_trades:
                    monitor_trade(symbol, df_with_indicators)
                else:
                    analyze_data(symbol, df_with_indicators)
            except Exception as e:
                logger.error(f"[{symbol}] Критическая ошибка в главном цикле: {e}", exc_info=True)
            time.sleep(1)
        logger.info(f"--- Итерация завершена. Следующая проверка через {MONITORING_INTERVAL_SECONDS} секунд. ---")
        time.sleep(MONITORING_INTERVAL_SECONDS)


if __name__ == "__main__":
    main_loop()
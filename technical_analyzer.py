# -*- coding: utf-8 -*-
import ccxt
import pandas as pd
import pandas_ta as ta
import time
# ИСПРАВЛЕНО: Правильный импорт для datetime.strptime
from datetime import datetime, timezone
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
warnings.filterwarnings(
    "ignore",
    message="Converting to PeriodArray/Index representation will drop timezone information.",
    category=UserWarning
)

# --- НОВАЯ ФУНКЦИЯ ДЛЯ ЗАГРУЗКИ ПОСЛЕДНЕЙ МОДЕЛИ (ИСПРАВЛЕНО) ---
def load_latest_model():
    model_dir = "models"
    pattern = "trade_model_*.pkl"
    files = glob.glob(os.path.join(model_dir, pattern))
    if not files:
        raise FileNotFoundError(f"В папке '{model_dir}' нет ни одного файла '{pattern}'")

    latest_file = None
    latest_timestamp_dt = None

    timestamp_pattern = re.compile(r"trade_model_(\d{8}_\d{4})\.pkl")

    for f in files:
        match = timestamp_pattern.search(f)
        if match:
            timestamp_str = match.group(1)
            try:
                current_timestamp_dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M")
                if latest_timestamp_dt is None or current_timestamp_dt > latest_timestamp_dt:
                    latest_timestamp_dt = current_timestamp_dt
                    latest_file = f
            except ValueError:
                logger.warning(f"Некорректный формат метки времени в имени файла: {f}")
                continue
    
    if latest_file:
        logger.info(f"Loading latest model: {latest_file}")
        mdl = joblib.load(latest_file) # ИСПРАВЛЕНО: использование latest_file
        return mdl["model"], mdl["features"]
    else:
        raise FileNotFoundError(f"Не удалось найти корректный файл модели в '{model_dir}'")

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

HISTORY_LIMIT = 500 

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

# --- НОВЫЕ НАСТРОЙКИ ДЛЯ ИНДИКАТОРОВ ---
# Настройки индикаторов
ATR_LENGTH = 14
RSI_LENGTH = 14
MACD_FAST_LENGTH = 12
MACD_SLOW_LENGTH = 26
MACD_SIGNAL_LENGTH = 9
BB_LENGTH = 20
BB_MULTIPLIER = 2.0
EMA_SHORT = 50
EMA_LONG = 200
VOLUME_EMA_LENGTH = 20
KAMA_LENGTH = 10
KAMA_FAST_EMA_PERIOD = 2
KAMA_SLOW_EMA_PERIOD = 30
CMF_LENGTH = 20
RVI_LENGTH = 14
STOCH_LENGTH = 14
STOCH_SMOOTH_K = 3
STOCH_SMOOTH_D = 3
ADX_LENGTH = 14 # <-- ДОБАВЬТЕ ЭТУ СТРОКУ
CCI_LENGTH = 20
VWAP_LENGTH = 14 # Хотя VWAP обычно не принимает length, добавим для полноты

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


def fetch_data(symbol: str) -> pd.DataFrame | None:
    try:
        ohlcv = EXCHANGE.fetch_ohlcv(symbol, TIMEFRAME, limit=5000) # Нужно больше для индикаторов
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True) # <-- ИСПРАВЛЕНО
        df.set_index('timestamp', inplace=True)
        return df
    except ccxt.NetworkError as e:
        logger.error(f"[{symbol}] Ошибка сети при получении данных: {e}")
        return None
    except Exception as e:
        logger.error(f"[{symbol}] Ошибка при получении данных OHLCV: {e}", exc_info=True)
        return None

    except ccxt.NetworkError as e:
        logger.error(f"[{symbol}] Ошибка сети при получении данных: {e}")
        return None
    except Exception as e:
        logger.error(f"[{symbol}] Неизвестная ошибка при получении данных: {e}", exc_info=True)
        return None


def calculate_atr_manually(df, length):
    """
    Ручной расчет Average True Range (ATR).
    Используется вместо pandas_ta.atr, чтобы избежать проблем с 'fillna'.
    """
    if df.empty:
        return df

    high_low = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift(1))
    low_prev_close = abs(df['low'] - df['close'].shift(1))

    true_range = pd.DataFrame({'hl': high_low, 'hpc': high_prev_close, 'lpc': low_prev_close}).max(axis=1)
    
    # Расчет EMA для ATR
    atr = true_range.ewm(span=length, adjust=False, min_periods=length).mean()
    df[f'ATR_{length}'] = atr
    return df


def add_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    # Убедимся, что timestamp является DatetimeIndex и в UTC
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None: # Если индекс наивный, но должен быть UTC
        df.index = df.index.tz_localize('UTC')
    elif df.index.tz != timezone.utc: # Если индекс с другим часовым поясом, конвертируем
        df.index = df.index.tz_convert('UTC')

    # Добавляем временные признаки (убеждаемся, что df.index уже UTC)
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month_of_year'] = df.index.month
    df['is_weekend'] = ((df.index.dayofweek == 5) | (df.index.dayofweek == 6)).astype(int)

    # --- Ваш код для добавления всех индикаторов (как в generate_history.py) ---
    # ATR
    df.ta.atr(length=14, append=True)
    # EMA
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    # RSI
    df.ta.rsi(length=14, append=True)
    # MACD
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    # Bollinger Bands
    df.ta.bbands(close=df['close'], length=20, std=2.0, append=True)
    df.rename(columns={
        f'BBL_20_2.0': 'BB_lower',
        f'BBM_20_2.0': 'BB_middle',
        f'BBU_20_2.0': 'BB_upper',
        f'BBB_20_2.0': 'BB_width'
    }, inplace=True)
    # VOLUME_EMA
    df.ta.ema(close=df['volume'], length=20, append=True, col_names=(f'VOLUME_EMA',))
    # KAMA
    df.ta.kama(close=df['close'], length=10, fast=2, slow=30, append=True)
    # CMF
    df.ta.cmf(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], length=20, append=True)
    # RVI
    df.ta.rvi(close=df['close'], length=14, append=True)
    # Stochastic Oscillator
    df.ta.stoch(high=df['high'], low=df['low'], close=df['close'], k=14, d=3, append=True)
    df.rename(columns={
        f'STOCHk_14_3_3': 'STOCH_k',
        f'STOCHd_14_3_3': 'STOCH_d'
    }, inplace=True)
    # ADX
    df.ta.adx(length=14, append=True)
    # CCI
    df.ta.cci(length=20, append=True)
    # VWAP
    if 'volume' in df.columns and not df['volume'].isnull().all():
        df.ta.vwap(append=True, fillna=True)


    # Удаляем строки с NaN значениями, которые появились из-за индикаторов
    # Список всех ожидаемых фичей, включая временные и переименованные BB/STOCH
    expected_features_generated = [ # Имена фичей, которые add_indicators ГЕНЕРИРУЕТ
        'hour_of_day', 'day_of_week', 'day_of_month', 'month_of_year', 'is_weekend',
        'ATR_14', 'EMA_50', 'EMA_200', 'RSI_14',
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'BB_upper', 'BB_middle', 'BB_lower', 'BB_width',
        'VOLUME_EMA', 'KAMA_10_2_30', 'CMF_20', 'RVI_14',
        'STOCH_k', 'STOCH_d', 'ADX_14', 'CCI_20'
    ]
    # VWAP_D - это дефолтное имя для VWAP в pandas_ta, если не указано другое
    if 'VWAP_D' in df.columns:
        expected_features_generated.append('VWAP_D')

    # Для `dropna` используем все столбцы, которые могли быть сгенерированы.
    required_for_dropna = [col for col in expected_features_generated if col in df.columns]
    
    initial_rows = len(df)
    df.dropna(subset=required_for_dropna, inplace=True)
    #if len(df) < initial_rows:
        #logger.warning(f"[{symbol}] Удалено {initial_rows - len(df)} строк из-за NaN в индикаторах.")

    return df

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
def filter_signal(symbol: str, side: str, df_with_indicators: pd.DataFrame) -> float:
    """
    Применяет ML-модель для фильтрации сигнала.
    Возвращает вероятность (prob) или 0.00, если фильтр отклонил сигнал.
    """
    global model, features # Доступ к глобальным переменным модели и признаков

    if model is None or features is None:
        logger.info(f"[{symbol}] ML-модель не загружена. Пропускаю ML-фильтр.")
        return 1.0 # Возвращаем 1.0, чтобы сигнал не был отклонен, если модель недоступна

    if df_with_indicators.empty:
        logger.warning(f"[{symbol}] Нет данных для ML-фильтра.")
        return 0.0

    # Извлекаем последнюю свечу для предсказания ИЗ ПОЛНОГО DATAFRAME
    current_candle_data = df_with_indicators.iloc[[-1]]

    # Создаем DataFrame для предсказания, гарантируя порядок и наличие всех признаков
    X_predict = pd.DataFrame(index=current_candle_data.index)
    
    for f in features: # Итерируемся по признакам, которые ожидает модель
        if f in current_candle_data.columns:
            X_predict[f] = current_candle_data[f]
        else:
            # Если признак, который ожидает модель, отсутствует в текущих данных
            logger.warning(f"[{symbol}] Отсутствует признак '{f}' для ML-модели. Возвращаю 0.00.")
            return 0.0 # Отклоняем сигнал, если нужной фичи нет.
            
    # Проверка на NaN/inf в X_predict (может случиться, если индикаторы не вычислились)
    if X_predict.isnull().any().any() or np.isinf(X_predict.values).any():
        logger.warning(f"[{symbol}] Обнаружены NaN или Inf в признаках для ML-модели. Возвращаю 0.00.")
        return 0.0 # Отклоняем сигнал в случае невалидных данных

    try:
        # Убедимся, что порядок колонок правильный перед предсказанием
        # predict_proba возвращает вероятности для обоих классов [prob_class_0, prob_class_1]
        # Нам нужна вероятность класса 1 (положительный сигнал)
        prob = model.predict_proba(X_predict[features])[:, 1][0] 
        return prob
    except Exception as e:
        logger.error(f"[{symbol}] Ошибка при предсказании ML-модели: {e}", exc_info=True)
        return 0.0 # Отклоняем сигнал в случае ошибки

def analyze_data(symbol, df):
    if len(df) < 2:
        logger.warning(f"[{symbol}] Недостаточно свечей ({len(df)}) для анализа после добавления индикаторов. Требуется минимум 2. Пропускаю анализ.")
        return # Выходим из функции, если данных недостаточно 
    """Анализирует данные и, если сигнал достаточно сильный, открывает сделку."""
    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    current_close_price = last_candle['close']

    long_score, short_score = 0, 0
    long_signal_reasons, short_signal_reasons = [], []
    signal_type_long, signal_type_short = "GENERIC_LONG", "GENERIC_SHORT" # Изменено для большей ясности

    # --- Существующие индикаторы (ВАША ЛОГИКА) ---
    # Убедитесь, что MACD_SUFFIX, KAMA_LENGTH, KAMA_FAST_EMA_PERIOD,
    # KAMA_SLOW_EMA_PERIOD, BBW_EMA_LENGTH, VOLUME_CONFIRMATION_MULTIPLIER, CMF_LENGTH
    # и другие константы определены глобально или передаются как аргументы.
    # Также убедитесь, что функции типа is_bollinger_bands_squeezing,
    # check_hammer_candlestick, check_engulfing_candlestick, calculate_dynamic_sl_tp, save_state
    # доступны в этом файле.

    # MACD
    if (prev_candle[f'MACD{MACD_SUFFIX}'] < prev_candle[f'MACDs{MACD_SUFFIX}'] and last_candle[f'MACD{MACD_SUFFIX}'] > last_candle[f'MACDs{MACD_SUFFIX}']):
        long_score += 1
        long_signal_reasons.append("MACD кроссовер вверх")
    if (prev_candle[f'MACD{MACD_SUFFIX}'] > prev_candle[f'MACDs{MACD_SUFFIX}'] and last_candle[f'MACD{MACD_SUFFIX}'] < last_candle[f'MACDs{MACD_SUFFIX}']):
        short_score += 1
        short_signal_reasons.append("MACD кроссовер вниз")
    
    # RSI
    if (prev_candle['RSI_14'] < 30 and last_candle['RSI_14'] > 30):
        long_score += 1
        long_signal_reasons.append(f"RSI выход из перепроданности ({last_candle['RSI_14']:.2f})")
    if (prev_candle['RSI_14'] > 70 and last_candle['RSI_14'] < 70):
        short_score += 1
        short_signal_reasons.append(f"RSI выход из перекупленности ({last_candle['RSI_14']:.2f})")
    
    # KAMA vs EMA
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

    # Bollinger Bands Width Squeeze
    bbw_col_name = 'BB_width' # Теперь используем переименованную колонку
    bbw_ema_col_name = f'BBW_EMA_{BBW_EMA_LENGTH}'
    if bbw_col_name in last_candle and bbw_ema_col_name in last_candle and is_bollinger_bands_squeezing(df, last_candle[bbw_col_name], last_candle[bbw_ema_col_name]):
        long_score += 1
        short_score += 1
        long_signal_reasons.append("BB Squeeze")
        short_signal_reasons.append("BB Squeeze")
    
    # VOLUME_EMA
    if last_candle['volume'] > (last_candle['VOLUME_EMA'] * VOLUME_CONFIRMATION_MULTIPLIER):
        long_score += 1
        short_score += 1
        long_signal_reasons.append("Повышенный объем")
        short_signal_reasons.append("Повышенный объем")

    # --- НОВЫЕ ИНДИКАТОРЫ И ЛОГИКА (ВАША ЛОГИКА) ---
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
    
    # --- Свечные паттерны (ВАША ЛОГИКА) ---
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
        logger.info(f"[{symbol}] Расчетные очки: LONG={long_score} (Причины: {'; '.join(long_signal_reasons)}), SHORT={short_score} (Причины: {'; '.join(short_signal_reasons)})")
        # ▶ ML-фильтр
        # ИСПРАВЛЕННЫЙ ВЫЗОВ ML-ФИЛЬТРА:
        # Передаем filter_signal полный DataFrame 'df', а не только last_candle.to_dict()
        # Это позволяет filter_signal получить все необходимые признаки, включая hour_of_day.
        prob = filter_signal(symbol, "LONG", df) 
        if prob < FILTER_THRESHOLD:
            logger.info(f"[{symbol}] ML-фильтр отклонил LONG (prob={prob:.2f} < {FILTER_THRESHOLD:.4f})")
            logger.info(f"[{symbol}] Расчетные очки: LONG={long_score} (Причины: {'; '.join(long_signal_reasons)}), SHORT={short_score} (Причины: {'; '.join(short_signal_reasons)})")
            return
        
        # --- Ваша логика открытия LONG сделки (остается неизменной) ---
        sl, tps = calculate_dynamic_sl_tp(current_close_price, df, "LONG", symbol, signal_type_long)
        active_trades[symbol] = {"type": "LONG", "entry_price": current_close_price, "sl": sl, "tp1": tps.get('TP1'), "tp2": tps.get('TP2'), "tp3": tps.get('TP3'), "status": "active"}
        logger.info(f"✅ [{symbol}] ОТКРЫТА LONG СДЕЛКА (Сила: {long_score}, ML-Prob: {prob:.2f}) @ {current_close_price:.8f}")
        logger.info(f"   SL: {sl:.8f}, TP1: {tps.get('TP1', 0.0):.8f}, TP2: {tps.get('TP2', 0.0):.8f}, TP3: {tps.get('TP3', 0.0):.8f}")
        logger.info(f"   Причины: {'; '.join(long_signal_reasons)}")
        save_state(active_trades) # Здесь save_state() вызывается без аргументов, как в вашем коде,
                     # если active_trades - глобальная переменная,
                     # или save_state(active_trades), если она ожидает аргумент.
        logger.info(f"[{symbol}] Расчетные очки: LONG={long_score} (Причины: {'; '.join(long_signal_reasons)}), SHORT={short_score} (Причины: {'; '.join(short_signal_reasons)})")
    elif short_score >= MIN_SIGNAL_STRENGTH:
        logger.info(f"[{symbol}] Расчетные очки: LONG={long_score} (Причины: {'; '.join(long_signal_reasons)}), SHORT={short_score} (Причины: {'; '.join(short_signal_reasons)})")
        # ▶ ML-фильтр
        # ИСПРАВЛЕННЫЙ ВЫЗОВ ML-ФИЛЬТРА:
        # Передаем filter_signal полный DataFrame 'df', а не только last_candle.to_dict()
        prob = filter_signal(symbol, "SHORT", df)
        if prob < FILTER_THRESHOLD:
            logger.info(f"[{symbol}] ML-фильтр отклонил SHORT (prob={prob:.2f} < {FILTER_THRESHOLD:.4f})")
            return

        # --- Ваша логика открытия SHORT сделки (остается неизменной) ---
        sl, tps = calculate_dynamic_sl_tp(current_close_price, df, "SHORT", symbol, signal_type_short)
        active_trades[symbol] = {"type": "SHORT", "entry_price": current_close_price, "sl": sl, "tp1": tps.get('TP1'), "tp2": tps.get('TP2'), "tp3": tps.get('TP3'), "status": "active"}
        logger.info(f"✅ [{symbol}] ОТКРЫТА SHORT СДЕЛКА (Сила: {short_score}, ML-Prob: {prob:.2f}) @ {current_close_price:.8f}")
        logger.info(f"   SL: {sl:.8f}, TP1: {tps.get('TP1', 0.0):.8f}, TP2: {tps.get('TP2', 0.0):.8f}, TP3: {tps.get('TP3', 0.0):.8f}")
        logger.info(f"   Причины: {'; '.join(short_signal_reasons)}")
        save_state(active_trades) # Здесь save_state() вызывается без аргументов, как в вашем коде.

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
    global model, features, EXCHANGE # Объявляем глобальные переменные, с которыми будем работать
    load_config()
    try:
        # --- ВОТ ЧТО НЕ ХВАТАЛО! Инициализация EXCHANGE ---
        EXCHANGE = ccxt.binance({
            'rateLimit': 1200,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future', # Это важно для торговли фьючерсами USDT-M
            }
        })
        EXCHANGE.load_markets()
        logger.info("✅ Биржа Binance USDT-M Futures успешно инициализирована.")
    except Exception as e:
        logger.critical(f"Ошибка при инициализации биржи: {e}", exc_info=True)
        exit() # Прерываем выполнение, если не удалось инициализировать биржу
        
    load_state() # Загружаем состояние активных сделок
    
    try:
        model, features = load_latest_model() # Загружаем ML-модель
        logger.info("✅ ML-модель успешно загружена.")
        if features:
            logger.info(f"Модель ожидает следующие признаки: {features}")
    except Exception as e:
        logger.critical(f"❌ Ошибка при загрузке ML-модели: {e}. Бот не будет использовать ML-фильтр.", exc_info=True)
        model = None # Сбрасываем модель и признаки, если загрузка не удалась
        features = None


def main_loop():
    """Основной цикл работы бота."""
    initialize_bot() # Здесь все еще должна быть ваша инициализация биржи и модели
    while True:
        logger.info(f"\n--- Новая итерация | Активных сделок: {len(active_trades)} ---")
        
        # ДОБАВЛЕНО: Проверка, что список символов не пуст
        if not SYMBOLS:
            logger.error("Список SYMBOLS пуст. Проверьте config.json на наличие символов и их корректную загрузку.")
            time.sleep(MONITORING_INTERVAL_SECONDS)
            continue

        for symbol in SYMBOLS:
            #logger.info(f"[{symbol}] Начинаю обработку символа.") # Новый лог: начало обработки символа
            
            df = fetch_data(symbol)
            if df is None or df.empty:
                logger.warning(f"[{symbol}] Не удалось получить данные или DataFrame пуст после fetch_data.") # Новый лог: данные не получены
                continue
            
            #logger.info(f"[{symbol}] Данные получены ({len(df)} свечей). Передаю для добавления индикаторов.") # Новый лог: данные получены
            df_with_indicators = add_indicators(df, symbol)
            
            if df_with_indicators.empty:
                logger.warning(f"[{symbol}] DataFrame пуст после добавления индикаторов.") # Новый лог: индикаторы не добавлены или очищены
                continue
            
            #logger.info(f"[{symbol}] Индикаторы добавлены. Передаю для анализа.") # Новый лог: индикаторы успешно добавлены
            try:
                if symbol in active_trades:
                    monitor_trade(symbol, df_with_indicators)
                else:
                    analyze_data(symbol, df_with_indicators)
            except Exception as e:
                logger.error(f"[{symbol}] Критическая ошибка при анализе/мониторинге: {e}", exc_info=True)
            
            # Важный момент: time.sleep(1) внутри цикла for symbol может сильно замедлять
            # итерацию, если у вас много символов. Если вы хотите сделать паузу между запросами
            # по каждому символу, 1 секунда может быть слишком много и приводить к RateLimit.
            # Если у вас всего 1 символ, то это не проблема.
            # Если символов много, рассмотрите удаление этого time.sleep(1) или его уменьшение.
            # time.sleep(1) # Это пауза между обработкой символов.
            
        logger.info(f"--- Итерация завершена. Следующая проверка через {MONITORING_INTERVAL_SECONDS} секунд. ---")
        time.sleep(MONITORING_INTERVAL_SECONDS) # Основная пауза между итерациями


if __name__ == "__main__":
    main_loop()
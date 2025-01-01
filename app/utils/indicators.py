import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

class TechnicalIndicators:
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data.rolling(window=period).mean()

    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(data: pd.Series, fast_period: int = 12, 
                      slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal Line"""
        fast_ema = TechnicalIndicators.calculate_ema(data, fast_period)
        slow_ema = TechnicalIndicators.calculate_ema(data, slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal_period)
        return macd_line, signal_line

    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, 
                                std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle_band = TechnicalIndicators.calculate_sma(data, period)
        std = data.rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        return upper_band, middle_band, lower_band

    @staticmethod
    def calculate_stochastic_oscillator(high: pd.Series, low: pd.Series, 
                                      close: pd.Series, k_period: int = 14, 
                                      d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_line = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_line = k_line.rolling(window=d_period).mean()
        return k_line, d_line

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def calculate_ichimoku(high: pd.Series, low: pd.Series, 
                          conversion_period: int = 9, 
                          base_period: int = 26, 
                          span_period: int = 52) -> Dict[str, pd.Series]:
        """Calculate Ichimoku Cloud components"""
        conversion_line = (high.rolling(window=conversion_period).max() + 
                         low.rolling(window=conversion_period).min()) / 2
        base_line = (high.rolling(window=base_period).max() + 
                    low.rolling(window=base_period).min()) / 2
        span_a = ((conversion_line + base_line) / 2).shift(base_period)
        span_b = ((high.rolling(window=span_period).max() + 
                  low.rolling(window=span_period).min()) / 2).shift(base_period)
        
        return {
            'conversion_line': conversion_line,
            'base_line': base_line,
            'span_a': span_a,
            'span_b': span_b
        }

    @staticmethod
    def calculate_fibonacci_retracement(high: float, low: float) -> Dict[str, float]:
        """Calculate Fibonacci Retracement Levels"""
        diff = high - low
        levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        retracements = {}
        
        for level in levels:
            retracements[f'level_{int(level * 1000)}'] = high - (diff * level)
        
        return retracements

    @staticmethod
    def calculate_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate Pivot Points and Support/Resistance Levels"""
        pivot = (high + low + close) / 3
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1,
            'r2': r2,
            'r3': r3,
            's1': s1,
            's2': s2,
            's3': s3
        }

class VolumeIndicators:
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        return (np.sign(close.diff()) * volume).cumsum()

    @staticmethod
    def calculate_vwap(high: pd.Series, low: pd.Series, 
                      close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()

    @staticmethod
    def calculate_cmf(high: pd.Series, low: pd.Series, 
                     close: pd.Series, volume: pd.Series, 
                     period: int = 20) -> pd.Series:
        """Calculate Chaikin Money Flow"""
        mfm = ((close - low) - (high - close)) / (high - low)
        mfv = mfm * volume
        return mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()

class VolatilityIndicators:
    @staticmethod
    def calculate_historical_volatility(close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Historical Volatility"""
        log_returns = np.log(close / close.shift(1))
        return log_returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
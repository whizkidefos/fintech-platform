import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class IndicatorType(Enum):
    TREND = "TREND"
    MOMENTUM = "MOMENTUM"
    VOLATILITY = "VOLATILITY"
    VOLUME = "VOLUME"
    CUSTOM = "CUSTOM"

@dataclass
class IndicatorParams:
    period: int
    signal_period: Optional[int] = None
    std_dev: Optional[float] = None
    k_period: Optional[int] = None
    d_period: Optional[int] = None
    roc_period: Optional[int] = None
    ma_type: str = 'sma'

class TechnicalIndicators:
    def __init__(self):
        self.indicators = {}

    def calculate_all(self, df: pd.DataFrame) -> Dict:
        """Calculate all basic indicators"""
        results = {}
        
        # Moving Averages
        results['sma'] = self.sma(df['close'], 20)
        results['ema'] = self.ema(df['close'], 20)
        
        # Trend Indicators
        results['macd'] = self.macd(df['close'])
        results['adx'] = self.adx(df['high'], df['low'], df['close'])
        
        # Momentum Indicators
        results['rsi'] = self.rsi(df['close'])
        results['stoch'] = self.stochastic(df['high'], df['low'], df['close'])
        results['cci'] = self.cci(df['high'], df['low'], df['close'])
        
        # Volatility Indicators
        results['bbands'] = self.bollinger_bands(df['close'])
        results['atr'] = self.atr(df['high'], df['low'], df['close'])
        
        # Volume Indicators
        results['obv'] = self.on_balance_volume(df['close'], df['volume'])
        results['mfi'] = self.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
        
        return results

    def sma(self, data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    def ema(self, data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    def macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Moving Average Convergence Divergence"""
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return {'macd': macd, 'signal': signal_line, 'histogram': histogram}

    def rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def bollinger_bands(self, data: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = self.sma(data, period)
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return {'middle': sma, 'upper': upper_band, 'lower': lower_band}

    def stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                  k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        return {'k': k, 'd': d}

    def adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        plus_dm = high.diff()
        minus_dm = low.diff()
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift()),
            'lc': abs(low - close.shift())
        }).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx

    def cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: pd.Series(x).mad())
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci

    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift()),
            'lc': abs(low - close.shift())
        }).max(axis=1)
        return tr.rolling(window=period).mean()

    def on_balance_volume(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        direction = np.where(close > close.shift(1), 1, -1)
        direction[0] = 0
        return (direction * volume).cumsum()

    def money_flow_index(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                        volume: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = pd.Series(0, index=money_flow.index)
        negative_flow = pd.Series(0, index=money_flow.index)
        
        # Calculate positive and negative money flow
        for i in range(1, len(typical_price)):
            if typical_price[i] > typical_price[i-1]:
                positive_flow[i] = money_flow[i]
            else:
                negative_flow[i] = money_flow[i]
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 * (positive_mf / (positive_mf + negative_mf))
        return mfi

    def ichimoku(self, high: pd.Series, low: pd.Series, 
                conversion_period: int = 9,
                base_period: int = 26,
                span_period: int = 52,
                displacement: int = 26) -> Dict[str, pd.Series]:
        """Ichimoku Cloud"""
        conversion = (high.rolling(window=conversion_period).max() + 
                     low.rolling(window=conversion_period).min()) / 2
        
        base = (high.rolling(window=base_period).max() + 
                low.rolling(window=base_period).min()) / 2
        
        span_a = ((conversion + base) / 2).shift(displacement)
        
        span_b = ((high.rolling(window=span_period).max() + 
                   low.rolling(window=span_period).min()) / 2).shift(displacement)
        
        return {
            'conversion': conversion,
            'base': base,
            'span_a': span_a,
            'span_b': span_b
        }

    def pivot_points(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """Calculate Pivot Points (Floor Trader's Method)"""
        pp = (high + low + close) / 3
        r1 = (2 * pp) - low
        s1 = (2 * pp) - high
        r2 = pp + (high - low)
        s2 = pp - (high - low)
        r3 = high + 2 * (pp - low)
        s3 = low - 2 * (high - pp)
        
        return {
            'pp': pp,
            'r1': r1, 's1': s1,
            'r2': r2, 's2': s2,
            'r3': r3, 's3': s3
        }

    def vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, 
             volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()

    def supertrend(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                  period: int = 10, multiplier: float = 3.0) -> Dict[str, pd.Series]:
        """SuperTrend Indicator"""
        atr = self.atr(high, low, close, period)
        
        upper_band = ((high + low) / 2) + (multiplier * atr)
        lower_band = ((high + low) / 2) - (multiplier * atr)
        
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)
        
        for i in range(period, len(close)):
            if close[i] > upper_band[i-1]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            elif close[i] < lower_band[i-1]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = supertrend[i-1]
                direction[i] = direction[i-1]
        
        return {
            'supertrend': supertrend,
            'direction': direction,
            'upper_band': upper_band,
            'lower_band': lower_band
        }
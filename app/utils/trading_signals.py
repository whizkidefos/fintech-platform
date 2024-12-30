import numpy as np
import pandas as pd
from datetime import datetime

class TechnicalIndicators:
    @staticmethod
    def calculate_sma(data, period):
        return data.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data, period):
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data, period=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
        fast_ema = TechnicalIndicators.calculate_ema(data, fast_period)
        slow_ema = TechnicalIndicators.calculate_ema(data, slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal_period)
        return macd_line, signal_line

class SignalGenerator:
    def __init__(self, asset_data):
        self.data = pd.DataFrame(asset_data)
        self.indicators = TechnicalIndicators()
        
    def generate_signals(self):
        signals = []
        
        # Calculate indicators
        self.data['SMA_20'] = self.indicators.calculate_sma(self.data['close'], 20)
        self.data['SMA_50'] = self.indicators.calculate_sma(self.data['close'], 50)
        self.data['RSI'] = self.indicators.calculate_rsi(self.data['close'])
        macd_line, signal_line = self.indicators.calculate_macd(self.data['close'])
        self.data['MACD'] = macd_line
        self.data['Signal_Line'] = signal_line
        
        # Generate signals based on multiple indicators
        for i in range(1, len(self.data)):
            signal = self._check_signal_conditions(i)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _check_signal_conditions(self, index):
        current = self.data.iloc[index]
        prev = self.data.iloc[index - 1]
        
        signal = {
            'timestamp': current.name,
            'price': current['close'],
            'indicators': {},
            'strength': 0,
            'type': None
        }
        
        # Check SMA crossover
        sma_crossover = (prev['SMA_20'] <= prev['SMA_50'] and 
                        current['SMA_20'] > current['SMA_50'])
        sma_crossunder = (prev['SMA_20'] >= prev['SMA_50'] and 
                         current['SMA_20'] < current['SMA_50'])
        
        # Check MACD crossover
        macd_crossover = (prev['MACD'] <= prev['Signal_Line'] and 
                         current['MACD'] > current['Signal_Line'])
        macd_crossunder = (prev['MACD'] >= prev['Signal_Line'] and 
                         current['MACD'] < current['Signal_Line'])
        
        # Check RSI conditions
        rsi_oversold = current['RSI'] < 30
        rsi_overbought = current['RSI'] > 70
        
        # Combine signals
        buy_signals = [sma_crossover, macd_crossover, rsi_oversold]
        sell_signals = [sma_crossunder, macd_crossunder, rsi_overbought]
        
        buy_strength = sum(buy_signals) / len(buy_signals)
        sell_strength = sum(sell_signals) / len(sell_signals)
        
        # Generate signal if strength threshold is met
        if buy_strength > 0.5:
            signal['type'] = 'buy'
            signal['strength'] = buy_strength
            signal['indicators'] = {
                'sma_crossover': sma_crossover,
                'macd_crossover': macd_crossover,
                'rsi_oversold': rsi_oversold
            }
            return signal
        elif sell_strength > 0.5:
            signal['type'] = 'sell'
            signal['strength'] = sell_strength
            signal['indicators'] = {
                'sma_crossunder': sma_crossunder,
                'macd_crossunder': macd_crossunder,
                'rsi_overbought': rsi_overbought
            }
            return signal
        
        return None
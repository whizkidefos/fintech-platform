from typing import Dict, List, Any
import pandas as pd
import numpy as np
from app.utils.indicators import TechnicalIndicators, VolumeIndicators, VolatilityIndicators

class SignalStrategy:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.indicators = TechnicalIndicators()
        self.volume_indicators = VolumeIndicators()
        self.volatility_indicators = VolatilityIndicators()
        self.signals = []

    def _check_volume_based_signals(self):
        """Check for volume-based signals"""
        avg_volume = self.data['volume'].rolling(window=20).mean()
        
        for i in range(1, len(self.data)):
            # Volume spike with price increase
            if (self.data['volume'].iloc[i] > 2 * avg_volume.iloc[i] and 
                self.data['close'].iloc[i] > self.data['close'].iloc[i-1]):
                self._add_signal(i, 'buy', 'Volume Spike', 0.5)
            
            # OBV trend
            if (self.data['obv'].iloc[i] > self.data['obv'].iloc[i-1] and 
                self.data['close'].iloc[i] < self.data['close'].iloc[i-1]):
                self._add_signal(i, 'buy', 'OBV Divergence', 0.6)

    def _check_ichimoku_signals(self):
        """Check for Ichimoku Cloud signals"""
        ichimoku = self.indicators.calculate_ichimoku(self.data['high'], self.data['low'])
        
        for i in range(26, len(self.data)):  # Start after cloud formation
            # Trend signals
            if (ichimoku['conversion_line'].iloc[i] > ichimoku['base_line'].iloc[i] and
                self.data['close'].iloc[i] > ichimoku['span_a'].iloc[i] and
                self.data['close'].iloc[i] > ichimoku['span_b'].iloc[i]):
                self._add_signal(i, 'buy', 'Ichimoku Bullish', 0.7)
            elif (ichimoku['conversion_line'].iloc[i] < ichimoku['base_line'].iloc[i] and
                  self.data['close'].iloc[i] < ichimoku['span_a'].iloc[i] and
                  self.data['close'].iloc[i] < ichimoku['span_b'].iloc[i]):
                self._add_signal(i, 'sell', 'Ichimoku Bearish', 0.7)

    def _add_signal(self, index: int, signal_type: str, strategy: str, strength: float):
        """Add a signal to the signals list"""
        signal = {
            'timestamp': self.data.index[index],
            'type': signal_type,
            'strategy': strategy,
            'strength': strength,
            'price': self.data['close'].iloc[index],
            'indicators': {
                'rsi': self.data['rsi'].iloc[index],
                'macd': self.data['macd'].iloc[index],
                'volume': self.data['volume'].iloc[index],
                'volatility': self.data['volatility'].iloc[index]
            }
        }
        self.signals.append(signal)

    def get_combined_signal(self, timeframe: str = '1h') -> Dict[str, Any]:
        """Generate a combined signal based on all strategies"""
        all_signals = self.generate_signals()
        if not all_signals:
            return None

        # Get recent signals within the timeframe
        recent_signals = [s for s in all_signals if s['timestamp'] > pd.Timestamp.now() - pd.Timedelta(timeframe)]
        
        if not recent_signals:
            return None

        # Calculate combined signal strength
        buy_strength = sum([s['strength'] for s in recent_signals if s['type'] == 'buy'])
        sell_strength = sum([s['strength'] for s in recent_signals if s['type'] == 'sell'])
        
        # Determine overall signal
        if buy_strength > sell_strength and buy_strength > 1.5:
            signal_type = 'buy'
            strength = buy_strength
        elif sell_strength > buy_strength and sell_strength > 1.5:
            signal_type = 'sell'
            strength = sell_strength
        else:
            signal_type = 'neutral'
            strength = 0

        return {
            'timestamp': pd.Timestamp.now(),
            'type': signal_type,
            'strength': strength,
            'signals_count': len(recent_signals),
            'contributing_signals': recent_signals
        }

class RiskManager:
    def __init__(self, portfolio_value: float, risk_per_trade: float = 0.02):
        self.portfolio_value = portfolio_value
        self.risk_per_trade = risk_per_trade

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> Dict[str, float]:
        """Calculate optimal position size based on risk parameters"""
        risk_amount = self.portfolio_value * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        position_size = risk_amount / price_risk
        
        return {
            'position_size': position_size,
            'risk_amount': risk_amount,
            'price_risk': price_risk,
            'max_loss': risk_amount
        }

    def adjust_for_correlation(self, position_size: float, correlation_matrix: pd.DataFrame) -> float:
        """Adjust position size based on correlation with existing positions"""
        # Implement correlation-based position sizing
        return position_size * 0.8  # Example adjustment

    def generate_stop_loss(self, entry_price: float, atr: float, 
                          multiplier: float = 2.0) -> Dict[str, float]:
        """Generate stop loss levels based on ATR"""
        stop_loss = entry_price - (atr * multiplier)
        return {
            'stop_loss': stop_loss,
            'risk_amount': entry_price - stop_loss
        }

    def generate_take_profit(self, entry_price: float, stop_loss: float, 
                           risk_reward_ratio: float = 2.0) -> float:
        """Generate take profit level based on risk-reward ratio"""
        risk = abs(entry_price - stop_loss)
        return entry_price + (risk * risk_reward_ratio)

class SignalExecutor:
    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager

    def prepare_order(self, signal: Dict[str, Any], current_price: float, 
                     atr: float) -> Dict[str, Any]:
        """Prepare order details based on signal and risk management"""
        # Generate stop loss
        stop_loss = self.risk_manager.generate_stop_loss(current_price, atr)
        
        # Calculate position size
        position_details = self.risk_manager.calculate_position_size(
            current_price, stop_loss['stop_loss']
        )
        
        # Generate take profit
        take_profit = self.risk_manager.generate_take_profit(
            current_price, stop_loss['stop_loss']
        )
        
        return {
            'signal_type': signal['type'],
            'entry_price': current_price,
            'stop_loss': stop_loss['stop_loss'],
            'take_profit': take_profit,
            'position_size': position_details['position_size'],
            'risk_amount': position_details['risk_amount'],
            'timestamp': pd.Timestamp.now()
        }
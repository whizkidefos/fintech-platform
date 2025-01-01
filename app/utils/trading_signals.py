import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from app.utils.indicators import TechnicalIndicators

@dataclass
class Signal:
    asset: str
    direction: str  # 'long' or 'short'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    timeframe: str
    strategy: str
    indicators: Dict
    metadata: Dict

class TradingSignalGenerator:
    def __init__(self, technical_indicators: TechnicalIndicators):
        self.indicators = technical_indicators
        self.scaler = StandardScaler()
        self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.regime_threshold = 0.5

    def generate_signals(self, market_data: Dict, regime_probabilities: Optional[np.ndarray] = None) -> List[Signal]:
        """Generate trading signals using multiple strategies"""
        signals = []
        
        # Technical Analysis Signals
        tech_signals = self._generate_technical_signals(market_data)
        signals.extend(tech_signals)
        
        # Statistical Arbitrage Signals
        stat_arb_signals = self._generate_stat_arb_signals(market_data)
        signals.extend(stat_arb_signals)
        
        # Machine Learning Signals
        ml_signals = self._generate_ml_signals(market_data)
        signals.extend(ml_signals)
        
        # Regime-based Signals
        if regime_probabilities is not None:
            regime_signals = self._generate_regime_signals(market_data, regime_probabilities)
            signals.extend(regime_signals)
        
        # Filter and rank signals
        ranked_signals = self._rank_signals(signals)
        
        return ranked_signals

    def _generate_technical_signals(self, market_data: Dict) -> List[Signal]:
        """Generate signals based on technical analysis"""
        signals = []
        
        for asset, data in market_data.items():
            # Calculate technical indicators
            indicators = self.indicators.calculate_all(data)
            
            # Trend following signals
            if (indicators['macd']['histogram'][-1] > 0 and
                indicators['adx'][-1] > 25):
                signals.append(Signal(
                    asset=asset,
                    direction='long',
                    confidence=0.7,
                    entry_price=data['close'][-1],
                    stop_loss=data['low'][-1] * 0.99,
                    take_profit=data['close'][-1] * 1.03,
                    timeframe='1h',
                    strategy='trend_following',
                    indicators=indicators,
                    metadata={'trend_strength': indicators['adx'][-1]}
                ))
            
            # Mean reversion signals
            if (indicators['rsi'][-1] < 30 and
                indicators['bbands']['lower'][-1] > data['close'][-1]):
                signals.append(Signal(
                    asset=asset,
                    direction='long',
                    confidence=0.6,
                    entry_price=data['close'][-1],
                    stop_loss=data['low'][-1] * 0.98,
                    take_profit=indicators['bbands']['middle'][-1],
                    timeframe='1h',
                    strategy='mean_reversion',
                    indicators=indicators,
                    metadata={'oversold_strength': 30 - indicators['rsi'][-1]}
                ))
        
        return signals

    def _generate_stat_arb_signals(self, market_data: Dict) -> List[Signal]:
        """Generate statistical arbitrage signals"""
        signals = []
        
        # Calculate correlation matrix
        returns = pd.DataFrame({asset: data['close'].pct_change()
                              for asset, data in market_data.items()})
        correlation = returns.corr()
        
        # Find highly correlated pairs
        for i in range(len(correlation)):
            for j in range(i + 1, len(correlation)):
                if correlation.iloc[i, j] > 0.8:
                    asset1 = correlation.index[i]
                    asset2 = correlation.index[j]
                    
                    # Calculate z-score of spread
                    spread = (market_data[asset1]['close'] /
                            market_data[asset2]['close'])
                    z_score = (spread - spread.mean()) / spread.std()
                    
                    if z_score[-1] > 2:
                        signals.extend([
                            Signal(
                                asset=asset1,
                                direction='short',
                                confidence=0.65,
                                entry_price=market_data[asset1]['close'][-1],
                                stop_loss=market_data[asset1]['close'][-1] * 1.02,
                                take_profit=market_data[asset1]['close'][-1] * 0.98,
                                timeframe='1h',
                                strategy='stat_arb',
                                indicators={},
                                metadata={'pair': asset2, 'z_score': float(z_score[-1])}
                            ),
                            Signal(
                                asset=asset2,
                                direction='long',
                                confidence=0.65,
                                entry_price=market_data[asset2]['close'][-1],
                                stop_loss=market_data[asset2]['close'][-1] * 0.98,
                                take_profit=market_data[asset2]['close'][-1] * 1.02,
                                timeframe='1h',
                                strategy='stat_arb',
                                indicators={},
                                metadata={'pair': asset1, 'z_score': float(z_score[-1])}
                            )
                        ])
        
        return signals

    def _generate_ml_signals(self, market_data: Dict) -> List[Signal]:
        """Generate machine learning based signals"""
        signals = []
        
        for asset, data in market_data.items():
            # Prepare features
            features = self._prepare_ml_features(data)
            
            # Make prediction
            if len(features) > 0:
                prediction = self.ml_model.predict_proba(features)[-1]
                confidence = max(prediction)
                
                if confidence > 0.7:
                    direction = 'long' if prediction[1] > prediction[0] else 'short'
                    signals.append(Signal(
                        asset=asset,
                        direction=direction,
                        confidence=confidence,
                        entry_price=data['close'][-1],
                        stop_loss=data['low'][-1] * 0.99,
                        take_profit=data['close'][-1] * 1.03,
                        timeframe='1h',
                        strategy='ml_prediction',
                        indicators={},
                        metadata={'model_confidence': float(confidence)}
                    ))
        
        return signals

    def _generate_regime_signals(self, market_data: Dict,
                               regime_probabilities: np.ndarray) -> List[Signal]:
        """Generate regime-dependent signals"""
        signals = []
        current_regime = np.argmax(regime_probabilities[-1])
        
        for asset, data in market_data.items():
            indicators = self.indicators.calculate_all(data)
            
            # High volatility regime
            if current_regime == 0:
                if indicators['atr'][-1] > indicators['atr'][-20:].mean():
                    signals.append(Signal(
                        asset=asset,
                        direction='long',
                        confidence=0.6,
                        entry_price=data['close'][-1],
                        stop_loss=data['close'][-1] - 2 * indicators['atr'][-1],
                        take_profit=data['close'][-1] + 3 * indicators['atr'][-1],
                        timeframe='1h',
                        strategy='regime_volatility',
                        indicators=indicators,
                        metadata={'regime': 'high_volatility'}
                    ))
            
            # Trend regime
            elif current_regime == 1:
                if all(indicators['sma'][-3:] > indicators['sma'][-4:-1]):
                    signals.append(Signal(
                        asset=asset,
                        direction='long',
                        confidence=0.7,
                        entry_price=data['close'][-1],
                        stop_loss=indicators['sma'][-1],
                        take_profit=data['close'][-1] * 1.05,
                        timeframe='1h',
                        strategy='regime_trend',
                        indicators=indicators,
                        metadata={'regime': 'trend'}
                    ))
            
            # Mean reversion regime
            else:
                if indicators['rsi'][-1] < 30:
                    signals.append(Signal(
                        asset=asset,
                        direction='long',
                        confidence=0.65,
                        entry_price=data['close'][-1],
                        stop_loss=data['low'][-1] * 0.98,
                        take_profit=indicators['bbands']['middle'][-1],
                        timeframe='1h',
                        strategy='regime_mean_reversion',
                        indicators=indicators,
                        metadata={'regime': 'mean_reversion'}
                    ))
        
        return signals

    def _prepare_ml_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for machine learning model"""
        # Calculate technical indicators
        indicators = self.indicators.calculate_all(data)
        
        features = pd.DataFrame({
            'rsi': indicators['rsi'],
            'macd': indicators['macd']['histogram'],
            'bb_position': (data['close'] - indicators['bbands']['middle']) /
                         (indicators['bbands']['upper'] - indicators['bbands']['middle']),
            'atr': indicators['atr'],
            'volume_sma_ratio': data['volume'] / data['volume'].rolling(20).mean(),
            'returns': data['close'].pct_change()
        }).dropna()
        
        return self.scaler.fit_transform(features)

    def _rank_signals(self, signals: List[Signal]) -> List[Signal]:
        """Rank and filter signals based on multiple criteria"""
        if not signals:
            return []
        
        # Calculate composite score for each signal
        scored_signals = []
        for signal in signals:
            score = signal.confidence
            
            # Adjust score based on strategy performance
            strategy_multiplier = {
                'trend_following': 1.2,
                'mean_reversion': 1.0,
                'stat_arb': 1.1,
                'ml_prediction': 1.0,
                'regime_volatility': 1.1,
                'regime_trend': 1.2,
                'regime_mean_reversion': 1.0
            }
            score *= strategy_multiplier.get(signal.strategy, 1.0)
            
            # Adjust score based on risk/reward
            risk = abs(signal.entry_price - signal.stop_loss)
            reward = abs(signal.take_profit - signal.entry_price)
            risk_reward_ratio = reward / risk if risk != 0 else 0
            score *= min(risk_reward_ratio, 3) / 3
            
            scored_signals.append((score, signal))
        
        # Sort by score and return top signals
        scored_signals.sort(key=lambda x: x[0], reverse=True)
        return [signal for _, signal in scored_signals[:10]]
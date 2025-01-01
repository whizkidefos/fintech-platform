from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from app.models import Asset, TradingSignal, Portfolio
from app.utils.market_data import MarketDataFetcher
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SignalConfig:
    strategy: str
    timeframe: str
    lookback: int
    threshold: float
    min_confidence: float
    max_signals: int

class SignalGenerator:
    def __init__(self):
        self.market_data = MarketDataFetcher()
        self.logger = logging.getLogger(__name__)
        
        # Default configurations for different strategies
        self.configs = {
            'trend_following': SignalConfig(
                strategy='trend_following',
                timeframe='1d',
                lookback=20,
                threshold=0.02,
                min_confidence=0.7,
                max_signals=5
            ),
            'mean_reversion': SignalConfig(
                strategy='mean_reversion',
                timeframe='1h',
                lookback=50,
                threshold=2.0,
                min_confidence=0.8,
                max_signals=3
            ),
            'breakout': SignalConfig(
                strategy='breakout',
                timeframe='1d',
                lookback=50,
                threshold=0.03,
                min_confidence=0.75,
                max_signals=3
            ),
            'momentum': SignalConfig(
                strategy='momentum',
                timeframe='1d',
                lookback=14,
                threshold=70,
                min_confidence=0.65,
                max_signals=5
            )
        }

    def generate_signals(self, portfolio: Portfolio) -> List[Dict]:
        """Generate trading signals for a portfolio"""
        try:
            signals = []
            
            # Get portfolio assets
            assets = [position.asset for position in portfolio.positions]
            
            # Add market indices and potential assets
            watchlist = self._get_watchlist(portfolio)
            assets.extend(watchlist)
            
            # Generate signals for each strategy
            for strategy, config in self.configs.items():
                strategy_signals = self._generate_strategy_signals(
                    assets, config
                )
                signals.extend(strategy_signals)
            
            # Filter and rank signals
            signals = self._filter_signals(signals, portfolio)
            signals = self._rank_signals(signals)
            
            # Create signal records
            self._create_signal_records(signals, portfolio)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return []

    def _generate_strategy_signals(self, assets: List[Asset],
                                 config: SignalConfig) -> List[Dict]:
        """Generate signals for a specific strategy"""
        signals = []
        
        for asset in assets:
            try:
                # Get historical data
                candles = self.market_data.get_candles(
                    asset.symbol,
                    timeframe=config.timeframe,
                    limit=config.lookback
                )
                
                if not candles:
                    continue
                
                # Convert to dataframe
                df = pd.DataFrame(candles)
                
                # Generate signal based on strategy
                if config.strategy == 'trend_following':
                    signal = self._trend_following_strategy(df, config)
                elif config.strategy == 'mean_reversion':
                    signal = self._mean_reversion_strategy(df, config)
                elif config.strategy == 'breakout':
                    signal = self._breakout_strategy(df, config)
                elif config.strategy == 'momentum':
                    signal = self._momentum_strategy(df, config)
                else:
                    continue
                
                if signal:
                    signal['asset'] = asset
                    signals.append(signal)
                    
            except Exception as e:
                self.logger.error(
                    f"Error generating {config.strategy} signals for "
                    f"{asset.symbol}: {str(e)}"
                )
                continue
        
        return signals

    def _trend_following_strategy(self, df: pd.DataFrame,
                                config: SignalConfig) -> Optional[Dict]:
        """Generate trend following signals"""
        try:
            # Calculate SMAs
            df['sma_short'] = df['close'].rolling(window=20).mean()
            df['sma_long'] = df['close'].rolling(window=50).mean()
            
            # Get latest values
            current_price = df['close'].iloc[-1]
            sma_short = df['sma_short'].iloc[-1]
            sma_long = df['sma_long'].iloc[-1]
            
            # Calculate signal
            if sma_short > sma_long * (1 + config.threshold):
                direction = 'long'
                strength = min((sma_short/sma_long - 1) / config.threshold, 1)
            elif sma_short < sma_long * (1 - config.threshold):
                direction = 'short'
                strength = min((1 - sma_short/sma_long) / config.threshold, 1)
            else:
                return None
            
            return {
                'strategy': config.strategy,
                'direction': direction,
                'strength': strength,
                'confidence': 0.8,
                'price': current_price,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error in trend following strategy: {str(e)}")
            return None

    def _mean_reversion_strategy(self, df: pd.DataFrame,
                               config: SignalConfig) -> Optional[Dict]:
        """Generate mean reversion signals"""
        try:
            # Calculate Bollinger Bands
            df['sma'] = df['close'].rolling(window=20).mean()
            df['std'] = df['close'].rolling(window=20).std()
            df['upper'] = df['sma'] + 2 * df['std']
            df['lower'] = df['sma'] - 2 * df['std']
            
            # Get latest values
            current_price = df['close'].iloc[-1]
            upper = df['upper'].iloc[-1]
            lower = df['lower'].iloc[-1]
            
            # Calculate signal
            if current_price > upper:
                direction = 'short'
                strength = min((current_price/upper - 1) * 5, 1)
            elif current_price < lower:
                direction = 'long'
                strength = min((1 - current_price/lower) * 5, 1)
            else:
                return None
            
            return {
                'strategy': config.strategy,
                'direction': direction,
                'strength': strength,
                'confidence': 0.7,
                'price': current_price,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error in mean reversion strategy: {str(e)}")
            return None

    def _breakout_strategy(self, df: pd.DataFrame,
                          config: SignalConfig) -> Optional[Dict]:
        """Generate breakout signals"""
        try:
            lookback = 20
            
            # Calculate support and resistance
            high_max = df['high'].rolling(window=lookback).max()
            low_min = df['low'].rolling(window=lookback).min()
            
            # Get latest values
            current_price = df['close'].iloc[-1]
            resistance = high_max.iloc[-2]  # Use previous period
            support = low_min.iloc[-2]
            
            # Calculate signal
            if current_price > resistance * (1 + config.threshold):
                direction = 'long'
                strength = min((current_price/resistance - 1) / config.threshold, 1)
            elif current_price < support * (1 - config.threshold):
                direction = 'short'
                strength = min((1 - current_price/support) / config.threshold, 1)
            else:
                return None
            
            return {
                'strategy': config.strategy,
                'direction': direction,
                'strength': strength,
                'confidence': 0.75,
                'price': current_price,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error in breakout strategy: {str(e)}")
            return None

    def _momentum_strategy(self, df: pd.DataFrame,
                         config: SignalConfig) -> Optional[Dict]:
        """Generate momentum signals"""
        try:
            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Get latest values
            current_price = df['close'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            
            # Calculate signal
            if rsi > 70:
                direction = 'short'
                strength = min((rsi - 70) / 30, 1)
            elif rsi < 30:
                direction = 'long'
                strength = min((30 - rsi) / 30, 1)
            else:
                return None
            
            return {
                'strategy': config.strategy,
                'direction': direction,
                'strength': strength,
                'confidence': 0.65,
                'price': current_price,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error in momentum strategy: {str(e)}")
            return None

    def _filter_signals(self, signals: List[Dict],
                       portfolio: Portfolio) -> List[Dict]:
        """Filter signals based on various criteria"""
        filtered = []
        
        for signal in signals:
            # Check confidence
            if signal['confidence'] < self.configs[signal['strategy']].min_confidence:
                continue
            
            # Check existing positions
            asset = signal['asset']
            position = next(
                (p for p in portfolio.positions if p.asset_id == asset.id),
                None
            )
            
            if position and signal['direction'] == 'long':
                continue
                
            if not position and signal['direction'] == 'short':
                continue
            
            filtered.append(signal)
        
        return filtered

    def _rank_signals(self, signals: List[Dict]) -> List[Dict]:
        """Rank signals by strength and confidence"""
        return sorted(
            signals,
            key=lambda x: x['strength'] * x['confidence'],
            reverse=True
        )

    def _create_signal_records(self, signals: List[Dict],
                             portfolio: Portfolio) -> None:
        """Create TradingSignal records in database"""
        try:
            for signal in signals:
                trading_signal = TradingSignal(
                    portfolio_id=portfolio.id,
                    asset_id=signal['asset'].id,
                    signal_type=signal['strategy'],
                    direction=signal['direction'],
                    strength=signal['strength'],
                    strategy=signal['strategy'],
                    confidence=signal['confidence'],
                    timestamp=signal['timestamp'],
                    expiration=signal['timestamp'] + timedelta(days=1)
                )
                db.session.add(trading_signal)
            
            db.session.commit()
            
        except Exception as e:
            self.logger.error(f"Error creating signal records: {str(e)}")
            db.session.rollback()

    def _get_watchlist(self, portfolio: Portfolio) -> List[Asset]:
        """Get watchlist assets"""
        # Implementation would get assets from user's watchlist
        # This is a simplified version
        indices = ['^GSPC', '^DJI', '^IXIC']  # S&P 500, Dow, Nasdaq
        watchlist = []
        
        for symbol in indices:
            try:
                asset = Asset.query.filter_by(symbol=symbol).first()
                if not asset:
                    ticker_data = self.market_data.get_ticker(symbol)
                    asset = Asset(
                        symbol=symbol,
                        name=ticker_data['name'],
                        asset_type='index',
                        current_price=ticker_data['price']
                    )
                    db.session.add(asset)
                watchlist.append(asset)
            except Exception as e:
                self.logger.error(
                    f"Error adding watchlist asset {symbol}: {str(e)}"
                )
                continue
        
        db.session.commit()
        return watchlist
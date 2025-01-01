import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from app.utils.signal_generator import SignalStrategy, RiskManager, SignalExecutor
from app.models import TradingSignal, Transaction, Asset, Portfolio
from app import db

logger = logging.getLogger(__name__)

class AutomatedTradingSystem:
    def __init__(self, 
                 exchange_client,
                 risk_manager: RiskManager,
                 signal_executor: SignalExecutor,
                 trading_pairs: List[str],
                 timeframes: List[str] = ['1m', '5m', '15m', '1h', '4h', '1d'],
                 max_open_trades: int = 5):
        self.exchange = exchange_client
        self.risk_manager = risk_manager
        self.signal_executor = signal_executor
        self.trading_pairs = trading_pairs
        self.timeframes = timeframes
        self.max_open_trades = max_open_trades
        self.active_trades = {}
        self.is_running = False
        self.last_update = {}

    async def _execute_trade(self, pair: str, signal: Dict[str, Any]) -> bool:
        """Execute a trade based on the signal"""
        try:
            # Get current market data
            ticker = await self.exchange.fetch_ticker(pair)
            current_price = ticker['last']
            
            # Prepare order parameters
            order_details = self.signal_executor.prepare_order(
                signal=signal,
                current_price=current_price,
                atr=self._calculate_atr(pair)
            )
            
            # Execute order
            order = await self.exchange.create_order(
                symbol=pair,
                type='market',
                side=signal['type'],
                amount=order_details['position_size']
            )
            
            if order['status'] == 'closed':
                # Record trade in active trades
                self.active_trades[pair] = {
                    'entry_price': order['average'],
                    'position_size': order_details['position_size'],
                    'stop_loss': order_details['stop_loss'],
                    'take_profit': order_details['take_profit'],
                    'trailing_stop': order_details['stop_loss'],
                    'type': signal['type'],
                    'entry_time': datetime.now(),
                    'order_id': order['id']
                }
                
                # Save to database
                await self._save_trade_to_db(pair, order, signal, order_details)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing trade for {pair}: {e}")
            return False

    async def _close_position(self, pair: str, reason: str) -> bool:
        """Close an open position"""
        try:
            trade = self.active_trades[pair]
            
            # Create closing order
            order = await self.exchange.create_order(
                symbol=pair,
                type='market',
                side='sell' if trade['type'] == 'buy' else 'buy',
                amount=trade['position_size']
            )
            
            if order['status'] == 'closed':
                # Calculate PnL
                entry_price = trade['entry_price']
                exit_price = order['average']
                pnl = (exit_price - entry_price) * trade['position_size'] if trade['type'] == 'buy' \
                    else (entry_price - exit_price) * trade['position_size']
                
                # Update database
                await self._update_trade_in_db(pair, order, pnl, reason)
                
                # Remove from active trades
                del self.active_trades[pair]
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error closing position for {pair}: {e}")
            return False

    async def _save_trade_to_db(self, pair: str, order: Dict, 
                               signal: Dict, order_details: Dict):
        """Save trade details to database"""
        try:
            # Create trading signal record
            signal_record = TradingSignal(
                asset_id=await self._get_asset_id(pair),
                signal_type=signal['type'],
                strength=signal['strength'],
                indicator_values=signal['indicators'],
                executed=True
            )
            db.session.add(signal_record)
            
            # Create transaction record
            transaction = Transaction(
                user_id=self.user_id,  # Set in initialization
                asset_id=await self._get_asset_id(pair),
                transaction_type=signal['type'],
                quantity=order_details['position_size'],
                price=order['average'],
                signal_id=signal_record.id
            )
            db.session.add(transaction)
            
            await db.session.commit()
            
        except Exception as e:
            logger.error(f"Error saving trade to database: {e}")
            await db.session.rollback()

    async def _update_trade_in_db(self, pair: str, order: Dict, pnl: float, reason: str):
        """Update trade records in database"""
        try:
            # Update transaction status
            transaction = Transaction.query.filter_by(
                asset_id=await self._get_asset_id(pair),
                status='open'
            ).first()
            
            if transaction:
                transaction.close_price = order['average']
                transaction.pnl = pnl
                transaction.close_reason = reason
                transaction.status = 'closed'
                transaction.closed_at = datetime.now()
                
                await db.session.commit()
                
        except Exception as e:
            logger.error(f"Error updating trade in database: {e}")
            await db.session.rollback()

    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate trading signal"""
        return (
            signal['strength'] >= 0.7 and  # Strong signal
            signal['type'] in ['buy', 'sell'] and  # Valid signal type
            all(i['value'] > 0 for i in signal['indicators'].values())  # Valid indicator values
        )

    def _can_take_new_position(self, pair: str) -> bool:
        """Check if we can take a new position"""
        return (
            len(self.active_trades) < self.max_open_trades and
            pair not in self.active_trades
        )

    def _check_stop_loss(self, trade: Dict[str, Any], current_price: float) -> bool:
        """Check if stop loss is hit"""
        if trade['type'] == 'buy':
            return current_price <= trade['stop_loss']
        return current_price >= trade['stop_loss']

    def _check_take_profit(self, trade: Dict[str, Any], current_price: float) -> bool:
        """Check if take profit is hit"""
        if trade['type'] == 'buy':
            return current_price >= trade['take_profit']
        return current_price <= trade['take_profit']

    def _adjust_trailing_stop(self, trade: Dict[str, Any], current_price: float):
        """Adjust trailing stop loss"""
        if trade['type'] == 'buy':
            potential_stop = current_price - (current_price * 0.02)  # 2% trailing stop
            if potential_stop > trade['trailing_stop']:
                trade['trailing_stop'] = potential_stop
                trade['stop_loss'] = potential_stop
        else:
            potential_stop = current_price + (current_price * 0.02)
            if potential_stop < trade['trailing_stop']:
                trade['trailing_stop'] = potential_stop
                trade['stop_loss'] = potential_stop

    async def _calculate_atr(self, pair: str) -> float:
        """Calculate Average True Range"""
        try:
            candles = await self.exchange.fetch_ohlcv(pair, '1h', limit=14)
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            true_ranges = []
            for i in range(1, len(df)):
                high_low = df['high'].iloc[i] - df['low'].iloc[i]
                high_close = abs(df['high'].iloc[i] - df['close'].iloc[i-1])
                low_close = abs(df['low'].iloc[i] - df['close'].iloc[i-1])
                true_ranges.append(max(high_low, high_close, low_close))
            
            return sum(true_ranges) / len(true_ranges)
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return None

    async def _get_asset_id(self, pair: str) -> int:
        """Get or create asset ID"""
        asset = await Asset.query.filter_by(symbol=pair).first()
        if not asset:
            asset = Asset(symbol=pair)
            db.session.add(asset)
            await db.session.commit()
        return asset.id
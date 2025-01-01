from typing import Dict, Optional, Union
from decimal import Decimal
import asyncio
from datetime import datetime
from app.models import Portfolio, Position, Transaction, TradingSignal
from app.utils.market_data import MarketDataFetcher
from app.utils.risk_manager import RiskManager

class OrderExecutor:
    def __init__(self, db_session, market_data: MarketDataFetcher):
        self.db = db_session
        self.market_data = market_data
        self.risk_manager = RiskManager()
        self.order_timeout = 30  # seconds
        self.max_slippage = Decimal('0.005')  # 0.5% maximum slippage

    async def execute_order(
        self,
        portfolio: Portfolio,
        signal: TradingSignal,
        position_size: Decimal
    ) -> Dict:
        """Execute the actual order"""
        try:
            # Get current market price
            ticker = await self.market_data.fetch_ticker(signal.asset.symbol)
            current_price = Decimal(str(ticker['last']))
            
            # Validate price slippage
            if not self._validate_slippage(current_price, Decimal(str(signal.price))):
                return {
                    'success': False,
                    'message': 'Price slippage too high'
                }
            
            # Calculate required margin and fees
            margin = self._calculate_margin(position_size, current_price)
            fees = self._calculate_fees(position_size, current_price)
            total_required = margin + fees
            
            # Check if sufficient funds
            if portfolio.cash_balance < total_required:
                return {
                    'success': False,
                    'message': 'Insufficient funds'
                }
            
            # Create position
            position = Position(
                portfolio_id=portfolio.id,
                asset_id=signal.asset_id,
                position_type=signal.signal_type,
                quantity=position_size,
                entry_price=current_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                leverage=self.risk_manager.get_leverage(portfolio),
                margin_used=margin
            )
            self.db.session.add(position)
            
            # Create transaction record
            transaction = Transaction(
                portfolio_id=portfolio.id,
                position_id=position.id,
                asset_id=signal.asset_id,
                transaction_type=signal.signal_type,
                quantity=position_size,
                price=current_price,
                fee=fees,
                status='completed',
                execution_id=f"signal_{signal.id}_{datetime.utcnow().timestamp()}"
            )
            self.db.session.add(transaction)
            
            # Update portfolio balance
            portfolio.cash_balance -= total_required
            
            await self.db.session.commit()
            
            return {
                'success': True,
                'position_id': position.id,
                'price': float(current_price),
                'quantity': float(position_size),
                'fees': float(fees)
            }
            
        except Exception as e:
            await self.db.session.rollback()
            return {
                'success': False,
                'message': f'Error executing order: {str(e)}'
            }

    def _validate_slippage(self, current_price: Decimal, signal_price: Decimal) -> bool:
        """Validate if price slippage is within acceptable range"""
        slippage = abs(current_price - signal_price) / signal_price
        return slippage <= self.max_slippage

    def _calculate_margin(self, size: Decimal, price: Decimal) -> Decimal:
        """Calculate required margin for position"""
        return (size * price) / self.risk_manager.get_leverage()

    def _calculate_fees(self, size: Decimal, price: Decimal) -> Decimal:
        """Calculate transaction fees"""
        base_fee_rate = Decimal('0.001')  # 0.1% base fee rate
        return size * price * base_fee_rate

    async def close_position(self, position_id: int) -> Dict:
        """Close an existing position"""
        try:
            position = await Position.query.get(position_id)
            if not position or position.status != 'open':
                return {
                    'success': False,
                    'message': 'Invalid or already closed position'
                }
            
            # Get current market price
            ticker = await self.market_data.fetch_ticker(position.asset.symbol)
            current_price = Decimal(str(ticker['last']))
            
            # Calculate P&L
            pnl = self._calculate_pnl(position, current_price)
            
            # Calculate fees
            fees = self._calculate_fees(position.quantity, current_price)
            
            # Create closing transaction
            transaction = Transaction(
                portfolio_id=position.portfolio_id,
                position_id=position.id,
                asset_id=position.asset_id,
                transaction_type='sell' if position.position_type == 'buy' else 'buy',
                quantity=position.quantity,
                price=current_price,
                fee=fees,
                status='completed'
            )
            self.db.session.add(transaction)
            
            # Update position
            position.status = 'closed'
            position.closed_at = datetime.utcnow()
            position.closing_price = current_price
            position.realized_pnl = pnl
            
            # Update portfolio balance
            position.portfolio.cash_balance += (position.margin_used + pnl - fees)
            
            await self.db.session.commit()
            
            return {
                'success': True,
                'pnl': float(pnl),
                'fees': float(fees),
                'closing_price': float(current_price)
            }
            
        except Exception as e:
            await self.db.session.rollback()
            return {
                'success': False,
                'message': f'Error closing position: {str(e)}'
            }

    def _calculate_pnl(self, position: Position, current_price: Decimal) -> Decimal:
        """Calculate realized P&L for position"""
        if position.position_type == 'buy':
            return (current_price - position.entry_price) * position.quantity
        else:
            return (position.entry_price - current_price) * position.quantity

class OrderManager:
    """Manage and monitor open orders and positions"""
    
    def __init__(self, db_session, market_data: MarketDataFetcher):
        self.db = db_session
        self.market_data = market_data
        self.executor = OrderExecutor(db_session, market_data)
        self.running = False

    async def start_monitoring(self):
        """Start monitoring open positions"""
        self.running = True
        while self.running:
            try:
                await self._check_open_positions()
                await asyncio.sleep(1)  # Check every second
            except Exception as e:
                print(f"Error monitoring positions: {e}")
                await asyncio.sleep(5)

    async def stop_monitoring(self):
        """Stop monitoring open positions"""
        self.running = False

    async def _check_open_positions(self):
        """Check all open positions for stop loss/take profit conditions"""
        positions = await Position.query.filter_by(status='open').all()
        
        for position in positions:
            try:
                ticker = await self.market_data.fetch_ticker(position.asset.symbol)
                current_price = Decimal(str(ticker['last']))
                
                if self._should_close_position(position, current_price):
                    await self.executor.close_position(position.id)
                    
            except Exception as e:
                print(f"Error checking position {position.id}: {e}")

    def _should_close_position(self, position: Position, current_price: Decimal) -> bool:
        """Check if position should be closed"""
        if position.position_type == 'buy':
            # Check stop loss
            if position.stop_loss and current_price <= position.stop_loss:
                return True
            # Check take profit
            if position.take_profit and current_price >= position.take_profit:
                return True
        else:  # short position
            # Check stop loss
            if position.stop_loss and current_price >= position.stop_loss:
                return True
            # Check take profit
            if position.take_profit and current_price <= position.take_profit:
                return True
        
        return False
from typing import Dict, Optional
from decimal import Decimal
import logging
from datetime import datetime
from app.models import Portfolio, Position, Transaction, Asset
from app.utils.risk_manager import RiskManager
from app.utils.market_data import MarketDataFetcher
from app import db

logger = logging.getLogger(__name__)

class OrderExecutor:
    def __init__(self):
        self.risk_manager = RiskManager()
        self.market_data = MarketDataFetcher()
        self.logger = logging.getLogger(__name__)

    def execute_order(self, portfolio_id: int, symbol: str, side: str,
                     quantity: Decimal, order_type: str,
                     price: Optional[Decimal] = None,
                     time_in_force: str = 'GTC') -> Dict:
        """Execute a trading order"""
        try:
            # Get portfolio and validate
            portfolio = Portfolio.query.get(portfolio_id)
            if not portfolio:
                raise ValueError("Portfolio not found")

            # Check risk limits
            risk_check, risk_message = self.risk_manager.check_order_risk(
                portfolio, symbol, float(quantity), side
            )
            if not risk_check:
                raise ValueError(f"Risk check failed: {risk_message}")

            # Get or create asset
            asset = Asset.query.filter_by(symbol=symbol).first()
            if not asset:
                asset = self._create_asset(symbol)

            # Get current market price if needed
            if not price or order_type == 'market':
                price = Decimal(str(self.market_data.get_ticker(symbol)['price']))

            # Execute the order
            if side == 'buy':
                position = self._handle_buy(
                    portfolio, asset, quantity, price
                )
            else:
                position = self._handle_sell(
                    portfolio, asset, quantity, price
                )

            # Create transaction record
            transaction = Transaction(
                portfolio_id=portfolio.id,
                asset_id=asset.id,
                transaction_type=side,
                quantity=float(quantity),
                price=float(price),
                status='completed'
            )
            db.session.add(transaction)
            db.session.commit()

            return {
                'status': 'completed',
                'position_id': position.id if position else None,
                'transaction_id': transaction.id,
                'executed_price': float(price),
                'executed_quantity': float(quantity)
            }

        except Exception as e:
            self.logger.error(f"Order execution error: {str(e)}")
            db.session.rollback()
            raise

    def close_position(self, position_id: int) -> Dict:
        """Close an existing position"""
        try:
            position = Position.query.get(position_id)
            if not position:
                raise ValueError("Position not found")

            # Get current market price
            current_price = Decimal(str(
                self.market_data.get_ticker(position.asset.symbol)['price']
            ))

            # Execute sell order
            result = self.execute_order(
                portfolio_id=position.portfolio_id,
                symbol=position.asset.symbol,
                side='sell',
                quantity=Decimal(str(position.quantity)),
                order_type='market',
                price=current_price
            )

            return {
                'status': 'completed',
                'position_id': position_id,
                'transaction_id': result['transaction_id'],
                'executed_price': float(current_price),
                'executed_quantity': float(position.quantity)
            }

        except Exception as e:
            self.logger.error(f"Position closure error: {str(e)}")
            db.session.rollback()
            raise

    def execute_signal(self, signal_id: int) -> Dict:
        """Execute a trading signal"""
        try:
            from app.models import TradingSignal
            signal = TradingSignal.query.get(signal_id)
            if not signal:
                raise ValueError("Signal not found")

            # Get current market price
            current_price = Decimal(str(
                self.market_data.get_ticker(signal.asset.symbol)['price']
            ))

            # Calculate position size based on signal strength
            portfolio = Portfolio.query.get(signal.portfolio_id)
            position_size = self._calculate_position_size(
                portfolio, signal, current_price
            )

            # Execute order based on signal direction
            result = self.execute_order(
                portfolio_id=signal.portfolio_id,
                symbol=signal.asset.symbol,
                side=signal.direction,
                quantity=position_size,
                order_type='market',
                price=current_price
            )

            return {
                'status': 'completed',
                'signal_id': signal_id,
                'transaction_id': result['transaction_id'],
                'executed_price': float(current_price),
                'executed_quantity': float(position_size)
            }

        except Exception as e:
            self.logger.error(f"Signal execution error: {str(e)}")
            db.session.rollback()
            raise

    def _create_asset(self, symbol: str) -> Asset:
        """Create a new asset record"""
        ticker_data = self.market_data.get_ticker(symbol)
        asset = Asset(
            symbol=symbol,
            name=ticker_data.get('name', symbol),
            asset_type=ticker_data.get('type', 'stock'),
            current_price=ticker_data['price']
        )
        db.session.add(asset)
        db.session.commit()
        return asset

    def _handle_buy(self, portfolio: Portfolio, asset: Asset,
                   quantity: Decimal, price: Decimal) -> Optional[Position]:
        """Handle buy order execution"""
        position = Position.query.filter_by(
            portfolio_id=portfolio.id,
            asset_id=asset.id
        ).first()

        if position:
            # Update existing position
            new_quantity = position.quantity + float(quantity)
            new_cost = (position.cost_basis * position.quantity +
                       float(price * quantity)) / new_quantity
            position.quantity = new_quantity
            position.cost_basis = new_cost
            position.last_updated = datetime.utcnow()
        else:
            # Create new position
            position = Position(
                portfolio_id=portfolio.id,
                asset_id=asset.id,
                quantity=float(quantity),
                cost_basis=float(price),
                entry_date=datetime.utcnow()
            )
            db.session.add(position)

        db.session.commit()
        return position

    def _handle_sell(self, portfolio: Portfolio, asset: Asset,
                    quantity: Decimal, price: Decimal) -> Optional[Position]:
        """Handle sell order execution"""
        position = Position.query.filter_by(
            portfolio_id=portfolio.id,
            asset_id=asset.id
        ).first()

        if not position:
            raise ValueError("No position found to sell")

        if position.quantity < float(quantity):
            raise ValueError("Insufficient position quantity")

        # Update position
        position.quantity -= float(quantity)
        position.last_updated = datetime.utcnow()

        # Remove position if fully closed
        if position.quantity == 0:
            db.session.delete(position)
            position = None

        db.session.commit()
        return position

    def _calculate_position_size(self, portfolio: Portfolio,
                               signal: 'TradingSignal',
                               current_price: Decimal) -> Decimal:
        """Calculate position size based on signal strength and portfolio value"""
        # Implementation would consider various factors
        # This is a simplified version
        base_size = portfolio.total_value * Decimal('0.1')  # 10% of portfolio
        adjusted_size = base_size * Decimal(str(signal.strength))
        return adjusted_size / current_price  # Convert to quantity
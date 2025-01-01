from app import db
from datetime import datetime
from decimal import Decimal

class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(64), nullable=False, default='Default Portfolio')
    description = db.Column(db.String(256))
    cash_balance = db.Column(db.Numeric(20, 8), nullable=False, default=0)
    equity_value = db.Column(db.Numeric(20, 8), nullable=False, default=0)
    realized_pnl = db.Column(db.Numeric(20, 8), nullable=False, default=0)
    unrealized_pnl = db.Column(db.Numeric(20, 8), nullable=False, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    positions = db.relationship('Position', backref='portfolio', lazy='dynamic')
    transactions = db.relationship('Transaction', backref='portfolio', lazy='dynamic')
    snapshots = db.relationship('PortfolioSnapshot', backref='portfolio', lazy='dynamic')

    def update_portfolio_value(self):
        """Update portfolio equity value and unrealized P&L"""
        from app.utils.market_data import MarketDataFetcher
        market_data = MarketDataFetcher()
        
        total_equity = Decimal('0')
        total_unrealized_pnl = Decimal('0')
        
        for position in self.positions.filter_by(status='open'):
            current_price = Decimal(str(market_data.fetch_ticker(position.asset.symbol)['last']))
            position_value = current_price * Decimal(str(position.quantity))
            
            # Calculate unrealized P&L
            if position.position_type == 'long':
                unrealized_pnl = position_value - (Decimal(str(position.entry_price)) * Decimal(str(position.quantity)))
            else:  # short position
                unrealized_pnl = (Decimal(str(position.entry_price)) * Decimal(str(position.quantity))) - position_value
            
            total_equity += position_value
            total_unrealized_pnl += unrealized_pnl
        
        # Add cash balance to total equity
        total_equity += self.cash_balance
        
        # Update portfolio values
        self.equity_value = total_equity
        self.unrealized_pnl = total_unrealized_pnl
        db.session.commit()

    def create_snapshot(self):
        """Create a portfolio value snapshot"""
        snapshot = PortfolioSnapshot(
            portfolio_id=self.id,
            cash_balance=self.cash_balance,
            equity_value=self.equity_value,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl
        )
        db.session.add(snapshot)
        db.session.commit()
        return snapshot

class Position(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolio.id'), nullable=False)
    asset_id = db.Column(db.Integer, db.ForeignKey('asset.id'), nullable=False)
    position_type = db.Column(db.String(10), nullable=False)  # 'long' or 'short'
    quantity = db.Column(db.Numeric(20, 8), nullable=False)
    entry_price = db.Column(db.Numeric(20, 8), nullable=False)
    current_price = db.Column(db.Numeric(20, 8))
    liquidation_price = db.Column(db.Numeric(20, 8))
    stop_loss = db.Column(db.Numeric(20, 8))
    take_profit = db.Column(db.Numeric(20, 8))
    leverage = db.Column(db.Numeric(5, 2), default=1)
    margin_used = db.Column(db.Numeric(20, 8))
    unrealized_pnl = db.Column(db.Numeric(20, 8), default=0)
    realized_pnl = db.Column(db.Numeric(20, 8), default=0)
    status = db.Column(db.String(20), default='open')  # open, closed, liquidated
    opened_at = db.Column(db.DateTime, default=datetime.utcnow)
    closed_at = db.Column(db.DateTime)
    closing_price = db.Column(db.Numeric(20, 8))
    
    # Relationships
    transactions = db.relationship('Transaction', backref='position', lazy='dynamic')

    def update_position_value(self):
        """Update position's current value and unrealized P&L"""
        from app.utils.market_data import MarketDataFetcher
        market_data = MarketDataFetcher()
        
        try:
            current_price = Decimal(str(market_data.fetch_ticker(self.asset.symbol)['last']))
            self.current_price = current_price
            
            position_value = current_price * Decimal(str(self.quantity))
            entry_value = Decimal(str(self.entry_price)) * Decimal(str(self.quantity))
            
            if self.position_type == 'long':
                self.unrealized_pnl = position_value - entry_value
            else:  # short position
                self.unrealized_pnl = entry_value - position_value
                
            # Check for liquidation
            if self.check_liquidation(current_price):
                self.liquidate()
            
            db.session.commit()
            
        except Exception as e:
            print(f"Error updating position value: {e}")
            db.session.rollback()

    def check_liquidation(self, current_price):
        """Check if position should be liquidated"""
        if not self.liquidation_price:
            return False
            
        if self.position_type == 'long':
            return current_price <= self.liquidation_price
        return current_price >= self.liquidation_price

    def liquidate(self):
        """Liquidate the position"""
        self.status = 'liquidated'
        self.closed_at = datetime.utcnow()
        self.closing_price = self.current_price
        self.realized_pnl = self.unrealized_pnl
        
        # Create liquidation transaction
        transaction = Transaction(
            portfolio_id=self.portfolio_id,
            position_id=self.id,
            asset_id=self.asset_id,
            transaction_type='liquidation',
            quantity=self.quantity,
            price=self.current_price,
            fee=0,  # Might want to include liquidation fees
            status='completed'
        )
        db.session.add(transaction)
        
        # Update portfolio
        self.portfolio.realized_pnl += self.realized_pnl
        self.portfolio.cash_balance += (self.margin_used + self.realized_pnl)
        
class PortfolioSnapshot(db.Model):
    """Historical portfolio value snapshots"""
    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolio.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    cash_balance = db.Column(db.Numeric(20, 8), nullable=False)
    equity_value = db.Column(db.Numeric(20, 8), nullable=False)
    realized_pnl = db.Column(db.Numeric(20, 8), nullable=False)
    unrealized_pnl = db.Column(db.Numeric(20, 8), nullable=False)
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'cash_balance': float(self.cash_balance),
            'equity_value': float(self.equity_value),
            'realized_pnl': float(self.realized_pnl),
            'unrealized_pnl': float(self.unrealized_pnl),
            'total_value': float(self.cash_balance + self.equity_value)
        }
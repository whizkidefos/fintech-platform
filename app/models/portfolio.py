from datetime import datetime
from app import db
from decimal import Decimal

class Portfolio(db.Model):
    __tablename__ = 'portfolios'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(64), nullable=False)
    description = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Risk profile (conservative, moderate, aggressive)
    risk_profile = db.Column(db.String(32), default='moderate')
    
    # Portfolio metrics
    total_value = db.Column(db.Numeric(20, 8), default=0)
    total_cost = db.Column(db.Numeric(20, 8), default=0)
    unrealized_pnl = db.Column(db.Numeric(20, 8), default=0)
    realized_pnl = db.Column(db.Numeric(20, 8), default=0)
    day_change = db.Column(db.Numeric(20, 8), default=0)
    day_change_pct = db.Column(db.Numeric(10, 4), default=0)
    
    # Relationships
    positions = db.relationship('Position', backref='portfolio', lazy=True)
    transactions = db.relationship('Transaction', backref='portfolio', lazy=True)
    trading_signals = db.relationship('TradingSignal', backref='portfolio', lazy=True)
    
    def __repr__(self):
        return f'<Portfolio {self.name}>'

class Position(db.Model):
    __tablename__ = 'positions'
    
    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolios.id'), nullable=False)
    asset_id = db.Column(db.Integer, db.ForeignKey('assets.id'), nullable=False)
    
    # Position details
    quantity = db.Column(db.Numeric(20, 8), nullable=False)
    cost_basis = db.Column(db.Numeric(20, 8), nullable=False)
    current_value = db.Column(db.Numeric(20, 8), nullable=False)
    unrealized_pnl = db.Column(db.Numeric(20, 8), default=0)
    realized_pnl = db.Column(db.Numeric(20, 8), default=0)
    return_pct = db.Column(db.Numeric(10, 4), default=0)
    
    # Position metrics
    day_change = db.Column(db.Numeric(20, 8), default=0)
    day_change_pct = db.Column(db.Numeric(10, 4), default=0)
    
    # Timestamps
    entry_date = db.Column(db.DateTime, default=datetime.utcnow)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Position {self.asset.symbol} {self.quantity}>'

class Transaction(db.Model):
    __tablename__ = 'transactions'
    
    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolios.id'), nullable=False)
    asset_id = db.Column(db.Integer, db.ForeignKey('assets.id'), nullable=False)
    
    # Transaction details
    transaction_type = db.Column(db.String(32), nullable=False)  # buy, sell
    quantity = db.Column(db.Numeric(20, 8), nullable=False)
    price = db.Column(db.Numeric(20, 8), nullable=False)
    total_amount = db.Column(db.Numeric(20, 8), nullable=False)
    fees = db.Column(db.Numeric(20, 8), default=0)
    
    # Order details
    order_type = db.Column(db.String(32), nullable=False)  # market, limit
    order_status = db.Column(db.String(32), nullable=False)  # filled, cancelled
    
    # Signal reference (if transaction was triggered by a signal)
    signal_id = db.Column(db.Integer, db.ForeignKey('trading_signals.id'))
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    executed_at = db.Column(db.DateTime)
    
    def __repr__(self):
        return f'<Transaction {self.transaction_type} {self.asset.symbol} {self.quantity}>'

class TradingSignal(db.Model):
    __tablename__ = 'trading_signals'
    
    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolios.id'), nullable=False)
    asset_id = db.Column(db.Integer, db.ForeignKey('assets.id'), nullable=False)
    
    # Signal details
    signal_type = db.Column(db.String(32), nullable=False)  # entry, exit
    direction = db.Column(db.String(32), nullable=False)  # long, short
    strength = db.Column(db.Float, nullable=False)  # 0 to 1
    confidence = db.Column(db.Float, nullable=False)  # 0 to 1
    
    # Signal metadata
    strategy = db.Column(db.String(64), nullable=False)
    timeframe = db.Column(db.String(32), nullable=False)
    status = db.Column(db.String(32), default='active')  # active, executed, expired
    executed = db.Column(db.Boolean, default=False)
    
    # Price levels
    entry_price = db.Column(db.Numeric(20, 8))
    stop_loss = db.Column(db.Numeric(20, 8))
    take_profit = db.Column(db.Numeric(20, 8))
    
    # Timestamps
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    expiration = db.Column(db.DateTime)
    executed_at = db.Column(db.DateTime)
    
    def __repr__(self):
        return f'<TradingSignal {self.signal_type} {self.asset.symbol} {self.direction}>'
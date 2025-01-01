from app import db
from datetime import datetime
from decimal import Decimal

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolio.id'), nullable=False)
    position_id = db.Column(db.Integer, db.ForeignKey('position.id'))
    asset_id = db.Column(db.Integer, db.ForeignKey('asset.id'), nullable=False)
    transaction_type = db.Column(db.String(20), nullable=False)  # buy, sell, deposit, withdrawal, liquidation
    quantity = db.Column(db.Numeric(20, 8), nullable=False)
    price = db.Column(db.Numeric(20, 8), nullable=False)
    fee = db.Column(db.Numeric(20, 8), default=0)
    total_amount = db.Column(db.Numeric(20, 8), nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, completed, failed, cancelled
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    executed_at = db.Column(db.DateTime)
    execution_id = db.Column(db.String(100))  # Exchange transaction ID
    notes = db.Column(db.String(500))

    def __init__(self, **kwargs):
        super(Transaction, self).__init__(**kwargs)
        self.calculate_total()

    def calculate_total(self):
        """Calculate total transaction amount including fees"""
        if hasattr(self, 'quantity') and hasattr(self, 'price'):
            quantity = Decimal(str(self.quantity))
            price = Decimal(str(self.price))
            fee = Decimal(str(self.fee)) if self.fee else Decimal('0')
            
            if self.transaction_type in ['buy', 'deposit']:
                self.total_amount = (quantity * price) + fee
            else:  # sell, withdrawal
                self.total_amount = (quantity * price) - fee

    def execute(self):
        """Execute the transaction"""
        try:
            if self.status != 'pending':
                raise ValueError(f"Transaction cannot be executed: status is {self.status}")

            # Update portfolio cash balance
            if self.transaction_type == 'buy':
                self.portfolio.cash_balance -= self.total_amount
            elif self.transaction_type == 'sell':
                self.portfolio.cash_balance += self.total_amount
            elif self.transaction_type == 'deposit':
                self.portfolio.cash_balance += self.quantity
            elif self.transaction_type == 'withdrawal':
                self.portfolio.cash_balance -= self.quantity

            self.status = 'completed'
            self.executed_at = datetime.utcnow()
            db.session.commit()

            # Create portfolio snapshot
            self.portfolio.create_snapshot()

        except Exception as e:
            db.session.rollback()
            self.status = 'failed'
            self.notes = str(e)
            db.session.commit()
            raise

    def to_dict(self):
        """Convert transaction to dictionary"""
        return {
            'id': self.id,
            'type': self.transaction_type,
            'asset': self.asset.symbol,
            'quantity': float(self.quantity),
            'price': float(self.price),
            'fee': float(self.fee),
            'total_amount': float(self.total_amount),
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'executed_at': self.executed_at.isoformat() if self.executed_at else None,
            'notes': self.notes
        }

class FeeSchedule(db.Model):
    """Trading fee schedule"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    tier = db.Column(db.String(20), nullable=False)
    maker_fee = db.Column(db.Numeric(10, 8), nullable=False)
    taker_fee = db.Column(db.Numeric(10, 8), nullable=False)
    withdrawal_fee = db.Column(db.Numeric(10, 8), nullable=False)
    min_trading_volume = db.Column(db.Numeric(20, 8), nullable=False)
    valid_from = db.Column(db.DateTime, nullable=False)
    valid_to = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    @classmethod
    def get_current_fees(cls, user_id):
        """Get current fee schedule for user"""
        return cls.query.filter(
            cls.user_id == user_id,
            cls.valid_from <= datetime.utcnow(),
            (cls.valid_to.is_(None) | (cls.valid_to >= datetime.utcnow()))
        ).first()

class TransactionLog(db.Model):
    """Detailed transaction log for audit purposes"""
    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(db.Integer, db.ForeignKey('transaction.id'), nullable=False)
    event_type = db.Column(db.String(50), nullable=False)
    details = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    @classmethod
    def log_event(cls, transaction_id, event_type, details=None):
        """Create a new transaction log entry"""
        log = cls(
            transaction_id=transaction_id,
            event_type=event_type,
            details=details
        )
        db.session.add(log)
        db.session.commit()
        return log
from datetime import datetime
from app import db
from decimal import Decimal
import json

class Asset(db.Model):
    __tablename__ = 'assets'
    
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(32), unique=True, nullable=False)
    name = db.Column(db.String(128))
    asset_type = db.Column(db.String(32))  # stock, crypto, forex, etc.
    
    # Current market data
    current_price = db.Column(db.Numeric(20, 8))
    day_open = db.Column(db.Numeric(20, 8))
    day_high = db.Column(db.Numeric(20, 8))
    day_low = db.Column(db.Numeric(20, 8))
    day_volume = db.Column(db.Numeric(20, 8))
    
    # Asset metadata (stored as JSON)
    asset_metadata = db.Column(db.Text)  # JSON string of additional metadata
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    positions = db.relationship('Position', backref='asset', lazy=True)
    transactions = db.relationship('Transaction', backref='asset', lazy=True)
    trading_signals = db.relationship('TradingSignal', backref='asset', lazy=True)
    
    def get_metadata(self):
        if self.asset_metadata:
            return json.loads(self.asset_metadata)
        return {}
    
    def set_metadata(self, metadata_dict):
        self.asset_metadata = json.dumps(metadata_dict)
    
    def update_price(self, price):
        """Update current price and calculate day metrics"""
        if not self.day_open:
            self.day_open = price
        self.current_price = price
        if not self.day_high or price > self.day_high:
            self.day_high = price
        if not self.day_low or price < self.day_low:
            self.day_low = price
    
    def __repr__(self):
        return f'<Asset {self.symbol}>'
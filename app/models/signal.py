from app import db
from datetime import datetime

class TradingSignal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    asset_id = db.Column(db.Integer, db.ForeignKey('asset.id'), nullable=False)
    signal_type = db.Column(db.String(20), nullable=False)  # buy, sell, hold
    strength = db.Column(db.Float)  # signal strength 0-1
    indicator_values = db.Column(db.JSON)  # Store indicator values that triggered the signal
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    executed = db.Column(db.Boolean, default=False)

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    asset_id = db.Column(db.Integer, db.ForeignKey('asset.id'), nullable=False)
    transaction_type = db.Column(db.String(20), nullable=False)  # buy, sell
    quantity = db.Column(db.Float, nullable=False)
    price = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    signal_id = db.Column(db.Integer, db.ForeignKey('trading_signal.id'))
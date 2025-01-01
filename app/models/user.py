from datetime import datetime
from app import db, login_manager
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
import json

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    
    # Profile information
    first_name = db.Column(db.String(64))
    last_name = db.Column(db.String(64))
    phone = db.Column(db.String(20))
    
    # User settings (stored as JSON)
    preferences = db.Column(db.Text)  # JSON string of user preferences
    notification_settings = db.Column(db.Text)  # JSON string of notification settings
    trading_settings = db.Column(db.Text)  # JSON string of trading preferences
    
    # Status and verification
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    portfolios = db.relationship('Portfolio', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def update_last_login(self):
        self.last_login = datetime.utcnow()
        db.session.commit()
    
    def get_preferences(self):
        if self.preferences:
            return json.loads(self.preferences)
        return {}
    
    def set_preferences(self, preferences_dict):
        self.preferences = json.dumps(preferences_dict)
    
    def get_notification_settings(self):
        if self.notification_settings:
            return json.loads(self.notification_settings)
        return {}
    
    def set_notification_settings(self, settings_dict):
        self.notification_settings = json.dumps(settings_dict)
    
    def get_trading_settings(self):
        if self.trading_settings:
            return json.loads(self.trading_settings)
        return {}
    
    def set_trading_settings(self, settings_dict):
        self.trading_settings = json.dumps(settings_dict)
    
    def __repr__(self):
        return f'<User {self.username}>'
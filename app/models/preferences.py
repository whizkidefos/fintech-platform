from app import db
from datetime import datetime
from typing import Dict, Any

class UserPreferences(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    theme = db.Column(db.String(20), default='light')
    default_timeframe = db.Column(db.String(20), default='1h')
    risk_level = db.Column(db.String(20), default='moderate')
    max_open_trades = db.Column(db.Integer, default=5)
    trading_pairs = db.Column(db.JSON, default=list)
    notifications_enabled = db.Column(db.Boolean, default=True)
    email_notifications = db.Column(db.Boolean, default=True)
    trading_enabled = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AlertRule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    asset_id = db.Column(db.Integer, db.ForeignKey('asset.id'), nullable=False)
    rule_type = db.Column(db.String(50), nullable=False)  # price, volume, indicator
    condition = db.Column(db.String(20), nullable=False)  # above, below, crosses
    value = db.Column(db.Float, nullable=False)
    timeframe = db.Column(db.String(20), default='1h')
    enabled = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_triggered = db.Column(db.DateTime)

class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    alert_rule_id = db.Column(db.Integer, db.ForeignKey('alert_rule.id'))
    asset_id = db.Column(db.Integer, db.ForeignKey('asset.id'), nullable=False)
    message = db.Column(db.String(500), nullable=False)
    priority = db.Column(db.String(20), default='medium')
    read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# app/utils/alert_manager.py
class AlertManager:
    def __init__(self, db_session):
        self.db = db_session

    async def check_alert_rules(self, asset_data: Dict[str, Any]):
        """Check if any alert rules are triggered"""
        try:
            rules = await AlertRule.query.filter_by(
                asset_id=asset_data['asset_id'],
                enabled=True
            ).all()

            for rule in rules:
                if self._check_rule_condition(rule, asset_data):
                    await self._trigger_alert(rule, asset_data)

        except Exception as e:
            logger.error(f"Error checking alert rules: {e}")

    def _check_rule_condition(self, rule: AlertRule, data: Dict[str, Any]) -> bool:
        """Check if alert rule condition is met"""
        try:
            current_value = self._get_value_for_rule(rule, data)
            
            if rule.condition == 'above':
                return current_value > rule.value
            elif rule.condition == 'below':
                return current_value < rule.value
            elif rule.condition == 'crosses':
                previous_value = self._get_previous_value(rule, data)
                return (previous_value < rule.value and current_value > rule.value) or \
                       (previous_value > rule.value and current_value < rule.value)
            
            return False

        except Exception as e:
            logger.error(f"Error checking rule condition: {e}")
            return False

    async def _trigger_alert(self, rule: AlertRule, data: Dict[str, Any]):
        """Create and send alert"""
        try:
            # Create alert record
            alert = Alert(
                user_id=rule.user_id,
                alert_rule_id=rule.id,
                asset_id=rule.asset_id,
                message=self._generate_alert_message(rule, data),
                priority=self._determine_priority(rule, data)
            )
            self.db.session.add(alert)
            
            # Update rule last triggered time
            rule.last_triggered = datetime.utcnow()
            
            await self.db.session.commit()
            
            # Send notifications based on user preferences
            await self._send_notifications(alert)

        except Exception as e:
            logger.error(f"Error triggering alert: {e}")
            await self.db.session.rollback()

    def _generate_alert_message(self, rule: AlertRule, data: Dict[str, Any]) -> str:
        """Generate alert message"""
        asset = Asset.query.get(rule.asset_id)
        current_value = self._get_value_for_rule(rule, data)
        
        return f"{asset.symbol} {rule.rule_type} {rule.condition} {rule.value} " \
               f"(Current: {current_value:.2f})"

    def _determine_priority(self, rule: AlertRule, data: Dict[str, Any]) -> str:
        """Determine alert priority"""
        current_value = self._get_value_for_rule(rule, data)
        if current_value > rule.value:
            return 'high'
        elif current_value < rule.value:
            return 'low'
        else:
            return 'medium'
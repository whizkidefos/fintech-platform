import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timedelta

class AlertManager:
    def __init__(self, db_session, config):
        self.db = db_session
        self.config = config
        self.notification_methods = {
            'email': self._send_email_notification,
            'web': self._send_web_notification,
            'webhook': self._send_webhook_notification
        }

    def _determine_priority(self, rule: AlertRule, data: Dict[str, Any]) -> str:
        """Determine alert priority based on rule type and market conditions"""
        current_value = self._get_value_for_rule(rule, data)
        threshold = abs(current_value - rule.value) / rule.value

        if rule.rule_type == 'price':
            if threshold > 0.1:  # 10% deviation
                return 'high'
            elif threshold > 0.05:  # 5% deviation
                return 'medium'
            return 'low'
        
        elif rule.rule_type == 'volume':
            if threshold > 1.0:  # 100% increase
                return 'high'
            elif threshold > 0.5:  # 50% increase
                return 'medium'
            return 'low'
        
        return 'medium'

    def _get_value_for_rule(self, rule: AlertRule, data: Dict[str, Any]) -> float:
        """Extract relevant value from data based on rule type"""
        if rule.rule_type == 'price':
            return data['price']
        elif rule.rule_type == 'volume':
            return data['volume']
        elif rule.rule_type == 'indicator':
            indicators = data.get('indicators', {})
            return indicators.get(rule.indicator_name, 0)
        return 0

    async def _send_notifications(self, alert: Alert):
        """Send notifications through configured channels"""
        user_prefs = await UserPreferences.query.filter_by(user_id=alert.user_id).first()
        
        if not user_prefs or not user_prefs.notifications_enabled:
            return

        if user_prefs.email_notifications:
            await self._send_email_notification(alert)
        
        await self._send_web_notification(alert)
        
        if hasattr(user_prefs, 'webhook_url') and user_prefs.webhook_url:
            await self._send_webhook_notification(alert, user_prefs.webhook_url)

    async def _send_email_notification(self, alert: Alert):
        """Send email notification"""
        try:
            user = await User.query.get(alert.user_id)
            asset = await Asset.query.get(alert.asset_id)

            msg = MIMEMultipart()
            msg['From'] = self.config.MAIL_DEFAULT_SENDER
            msg['To'] = user.email
            msg['Subject'] = f"Fintech Platform Alert: {asset.symbol}"

            body = self._generate_email_body(alert)
            msg.attach(MIMEText(body, 'html'))

            with smtplib.SMTP(self.config.MAIL_SERVER, self.config.MAIL_PORT) as server:
                if self.config.MAIL_USE_TLS:
                    server.starttls()
                if self.config.MAIL_USERNAME:
                    server.login(self.config.MAIL_USERNAME, self.config.MAIL_PASSWORD)
                server.send_message(msg)

        except Exception as e:
            logger.error(f"Error sending email notification: {e}")

    def _generate_email_body(self, alert: Alert) -> str:
        """Generate HTML email body for alert"""
        template = """
        <html>
            <body style="font-family: Arial, sans-serif;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #2563eb;">Fintech Platform Alert</h2>
                    <div style="background-color: #f3f4f6; padding: 15px; border-radius: 5px;">
                        <p><strong>Alert Type:</strong> {type}</p>
                        <p><strong>Asset:</strong> {asset}</p>
                        <p><strong>Message:</strong> {message}</p>
                        <p><strong>Time:</strong> {time}</p>
                    </div>
                    <div style="margin-top: 20px; font-size: 12px; color: #6b7280;">
                        <p>This is an automated message from Fintech Platform.</p>
                    </div>
                </div>
            </body>
        </html>
        """
        
        return template.format(
            type=alert.alert_rule.rule_type.title(),
            asset=alert.asset.symbol,
            message=alert.message,
            time=alert.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        )

    async def _send_web_notification(self, alert: Alert):
        """Send web notification through WebSocket"""
        try:
            from app.websocket import websocket_manager
            
            notification_data = {
                'type': 'alert',
                'data': {
                    'id': alert.id,
                    'message': alert.message,
                    'priority': alert.priority,
                    'asset': alert.asset.symbol,
                    'timestamp': alert.created_at.isoformat()
                }
            }
            
            await websocket_manager.send_to_user(
                user_id=alert.user_id,
                message=notification_data
            )

        except Exception as e:
            logger.error(f"Error sending web notification: {e}")

    async def _send_webhook_notification(self, alert: Alert, webhook_url: str):
        """Send notification to configured webhook"""
        try:
            import aiohttp
            
            payload = {
                'id': alert.id,
                'type': alert.alert_rule.rule_type,
                'asset': alert.asset.symbol,
                'message': alert.message,
                'priority': alert.priority,
                'timestamp': alert.created_at.isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status not in (200, 201):
                        logger.error(f"Webhook notification failed with status {response.status}")

        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")

    async def cleanup_old_alerts(self, days: int = 30):
        """Clean up old alerts from the database"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            await Alert.query.filter(
                Alert.created_at < cutoff_date,
                Alert.read == True
            ).delete()
            
            await self.db.session.commit()

        except Exception as e:
            logger.error(f"Error cleaning up old alerts: {e}")
            await self.db.session.rollback()

    async def mark_alerts_read(self, user_id: int, alert_ids: List[int]):
        """Mark multiple alerts as read"""
        try:
            await Alert.query.filter(
                Alert.user_id == user_id,
                Alert.id.in_(alert_ids)
            ).update({
                Alert.read: True
            }, synchronize_session=False)
            
            await self.db.session.commit()

        except Exception as e:
            logger.error(f"Error marking alerts as read: {e}")
            await self.db.session.rollback()
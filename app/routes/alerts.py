from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from app.models import AlertRule, Alert, UserPreferences
from app.utils.alert_manager import AlertManager
from app import db

bp = Blueprint('alerts', __name__, url_prefix='/api/alerts')

@bp.route('/rules', methods=['GET'])
@login_required
def get_alert_rules():
    """Get all alert rules for current user"""
    rules = AlertRule.query.filter_by(user_id=current_user.id).all()
    return jsonify({
        'status': 'success',
        'data': [{
            'id': rule.id,
            'asset_id': rule.asset_id,
            'rule_type': rule.rule_type,
            'condition': rule.condition,
            'value': rule.value,
            'enabled': rule.enabled,
            'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None
        } for rule in rules]
    })

@bp.route('/rules', methods=['POST'])
@login_required
def create_alert_rule():
    """Create new alert rule"""
    data = request.get_json()
    
    try:
        rule = AlertRule(
            user_id=current_user.id,
            asset_id=data['asset_id'],
            rule_type=data['rule_type'],
            condition=data['condition'],
            value=data['value'],
            timeframe=data.get('timeframe', '1h')
        )
        
        db.session.add(rule)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Alert rule created successfully',
            'data': {
                'id': rule.id
            }
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@bp.route('/rules/<int:rule_id>', methods=['DELETE'])
@login_required
def delete_alert_rule(rule_id):
    """Delete alert rule"""
    rule = AlertRule.query.filter_by(
        id=rule_id,
        user_id=current_user.id
    ).first_or_404()
    
    try:
        db.session.delete(rule)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Alert rule deleted successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@bp.route('/notifications', methods=['GET'])
@login_required
def get_alerts():
    """Get user alerts"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    alerts = Alert.query.filter_by(
        user_id=current_user.id
    ).order_by(
        Alert.created_at.desc()
    ).paginate(page=page, per_page=per_page)
    
    return jsonify({
        'status': 'success',
        'data': {
            'alerts': [{
                'id': alert.id,
                'message': alert.message,
                'priority': alert.priority,
                'read': alert.read,
                'created_at': alert.created_at.isoformat()
            } for alert in alerts.items],
            'pagination': {
                'total': alerts.total,
                'pages': alerts.pages,
                'current_page': alerts.page,
                'per_page': alerts.per_page
            }
        }
    })

@bp.route('/notifications/mark-read', methods=['POST'])
@login_required
def mark_alerts_read():
    """Mark alerts as read"""
    data = request.get_json()
    alert_ids = data.get('alert_ids', [])
    
    try:
        Alert.query.filter(
            Alert.id.in_(alert_ids),
            Alert.user_id == current_user.id
        ).update({
            Alert.read: True
        }, synchronize_session=False)
        
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Alerts marked as read'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@bp.route('/preferences', methods=['GET'])
@login_required
def get_preferences():
    """Get user preferences"""
    prefs = UserPreferences.query.filter_by(user_id=current_user.id).first()
    
    if not prefs:
        prefs = UserPreferences(user_id=current_user.id)
        db.session.add(prefs)
        db.session.commit()
    
    return jsonify({
        'status': 'success',
        'data': {
            'theme': prefs.theme,
            'default_timeframe': prefs.default_timeframe,
            'risk_level': prefs.risk_level,
            'max_open_trades': prefs.max_open_trades,
            'trading_pairs': prefs.trading_pairs,
            'notifications_enabled': prefs.notifications_enabled,
            'email_notifications': prefs.email_notifications,
            'trading_enabled': prefs.trading_enabled
        }
    })

@bp.route('/preferences', methods=['PUT'])
@login_required
def update_preferences():
    """Update user preferences"""
    data = request.get_json()
    
    try:
        prefs = UserPreferences.query.filter_by(user_id=current_user.id).first()
        
        if not prefs:
            prefs = UserPreferences(user_id=current_user.id)
            db.session.add(prefs)
        
        # Update fields
        for key, value in data.items():
            if hasattr(prefs, key):
                setattr(prefs, key, value)
        
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Preferences updated successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
from flask import Blueprint, render_template, jsonify, request
from flask_login import login_required, current_user
from app.models import Portfolio, Position, TradingSignal, Asset, Transaction
from app.utils.market_data import MarketDataFetcher
from app import db
from datetime import datetime, timedelta
import pandas as pd

bp = Blueprint('dashboard', __name__)
market_data = MarketDataFetcher()

@bp.route('/')
@bp.route('/dashboard')
@login_required
def index():
    """Main dashboard page"""
    # Get portfolio data
    portfolio = get_portfolio_summary(current_user.id)
    
    # Get active positions
    positions = get_active_positions(current_user.id)
    
    # Get recent signals
    signals = get_recent_signals()
    
    # Get market overview data
    market_data = get_market_overview()
    
    return render_template('dashboard/index.html',
                         portfolio=portfolio,
                         positions=positions,
                         signals=signals,
                         market_data=market_data)

def get_portfolio_summary(user_id):
    """Get portfolio summary including value and changes"""
    try:
        portfolio = Portfolio.query.filter_by(user_id=user_id).first()
        if not portfolio:
            return create_default_portfolio_summary()

        # Calculate total value and changes
        total_value = calculate_portfolio_value(portfolio)
        daily_change = calculate_daily_change(portfolio)
        daily_pnl = calculate_daily_pnl(portfolio)
        open_positions = Position.query.filter_by(
            portfolio_id=portfolio.id, 
            status='open'
        ).count()

        return {
            'total_value': total_value,
            'daily_change': daily_change,
            'daily_pnl': daily_pnl,
            'open_positions': open_positions,
        }
    except Exception as e:
        print(f"Error getting portfolio summary: {e}")
        return create_default_portfolio_summary()

def calculate_portfolio_value(portfolio):
    """Calculate current total portfolio value"""
    total_value = 0
    try:
        # Get all positions
        positions = Position.query.filter_by(
            portfolio_id=portfolio.id,
            status='open'
        ).all()

        # Calculate positions value
        for position in positions:
            current_price = market_data.fetch_ticker(position.asset.symbol)['last']
            position_value = position.quantity * current_price
            total_value += position_value

        # Add cash balance
        total_value += portfolio.cash_balance
        return total_value
    except Exception as e:
        print(f"Error calculating portfolio value: {e}")
        return 0

def calculate_daily_change(portfolio):
    """Calculate 24h portfolio change percentage"""
    try:
        # Get yesterday's portfolio value
        yesterday = datetime.utcnow() - timedelta(days=1)
        yesterday_value = get_historical_portfolio_value(portfolio.id, yesterday)
        
        if yesterday_value == 0:
            return 0

        current_value = calculate_portfolio_value(portfolio)
        return ((current_value - yesterday_value) / yesterday_value) * 100
    except Exception as e:
        print(f"Error calculating daily change: {e}")
        return 0

def get_active_positions(user_id):
    """Get all active positions with current P&L"""
    try:
        positions = Position.query.join(Portfolio).filter(
            Portfolio.user_id == user_id,
            Position.status == 'open'
        ).all()

        position_data = []
        for position in positions:
            current_price = market_data.fetch_ticker(position.asset.symbol)['last']
            pnl = calculate_position_pnl(position, current_price)
            
            position_data.append({
                'id': position.id,
                'symbol': position.asset.symbol,
                'type': position.position_type,
                'entry_price': position.entry_price,
                'current_price': current_price,
                'size': position.quantity,
                'pnl': pnl['value'],
                'pnl_percent': pnl['percentage']
            })

        return position_data
    except Exception as e:
        print(f"Error getting active positions: {e}")
        return []

def calculate_position_pnl(position, current_price):
    """Calculate P&L for a position"""
    try:
        if position.position_type == 'long':
            pnl_value = (current_price - position.entry_price) * position.quantity
        else:  # short position
            pnl_value = (position.entry_price - current_price) * position.quantity

        pnl_percentage = (pnl_value / (position.entry_price * position.quantity)) * 100
        
        return {
            'value': pnl_value,
            'percentage': pnl_percentage
        }
    except Exception as e:
        print(f"Error calculating position P&L: {e}")
        return {'value': 0, 'percentage': 0}

def get_recent_signals():
    """Get recent trading signals"""
    try:
        return TradingSignal.query.order_by(
            TradingSignal.created_at.desc()
        ).limit(10).all()
    except Exception as e:
        print(f"Error getting recent signals: {e}")
        return []

def get_market_overview():
    """Get market overview data for major assets"""
    try:
        assets = Asset.query.filter_by(is_active=True).all()
        market_data = []
        
        for asset in assets:
            ticker = market_data.fetch_ticker(asset.symbol)
            market_data.append({
                'symbol': asset.symbol,
                'name': asset.name,
                'price': ticker['last'],
                'change': ticker['percentage'],
                'volume': ticker['baseVolume']
            })

        return market_data
    except Exception as e:
        print(f"Error getting market overview: {e}")
        return []

@bp.route('/api/portfolio/history')
@login_required
def portfolio_history():
    """Get portfolio value history"""
    try:
        timeframe = request.args.get('timeframe', '1d')
        data = get_portfolio_history(current_user.id, timeframe)
        return jsonify({
            'status': 'success',
            'data': data
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@bp.route('/api/position/close', methods=['POST'])
@login_required
def close_position():
    """Close a position"""
    try:
        data = request.get_json()
        position_id = data.get('position_id')
        
        position = Position.query.filter_by(
            id=position_id,
            portfolio_id=current_user.portfolio.id
        ).first_or_404()

        # Get current market price
        current_price = market_data.fetch_ticker(position.asset.symbol)['last']

        # Create closing transaction
        transaction = Transaction(
            portfolio_id=position.portfolio_id,
            asset_id=position.asset_id,
            transaction_type='sell' if position.position_type == 'long' else 'buy',
            quantity=position.quantity,
            price=current_price,
            status='completed'
        )
        db.session.add(transaction)

        # Update position status
        position.status = 'closed'
        position.closed_at = datetime.utcnow()
        position.closing_price = current_price

        # Calculate and update P&L
        pnl = calculate_position_pnl(position, current_price)
        position.realized_pnl = pnl['value']

        # Update portfolio cash balance
        portfolio = position.portfolio
        portfolio.cash_balance += (position.quantity * current_price)

        db.session.commit()

        return jsonify({
            'status': 'success',
            'message': 'Position closed successfully',
            'data': {
                'pnl': pnl['value'],
                'pnl_percentage': pnl['percentage']
            }
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@bp.route('/api/position/new', methods=['POST'])
@login_required
def open_position():
    """Open a new position"""
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['symbol', 'type', 'quantity', 'price']
        if not all(field in data for field in required_fields):
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields'
            }), 400

        # Check if asset exists
        asset = Asset.query.filter_by(symbol=data['symbol']).first_or_404()

        # Calculate required capital
        required_capital = data['quantity'] * data['price']

        # Check if sufficient funds
        portfolio = Portfolio.query.filter_by(user_id=current_user.id).first()
        if portfolio.cash_balance < required_capital:
            return jsonify({
                'status': 'error',
                'message': 'Insufficient funds'
            }), 400

        # Create new position
        position = Position(
            portfolio_id=portfolio.id,
            asset_id=asset.id,
            position_type=data['type'],
            quantity=data['quantity'],
            entry_price=data['price'],
            status='open'
        )
        db.session.add(position)

        # Create transaction record
        transaction = Transaction(
            portfolio_id=portfolio.id,
            asset_id=asset.id,
            transaction_type='buy' if data['type'] == 'long' else 'sell',
            quantity=data['quantity'],
            price=data['price'],
            status='completed'
        )
        db.session.add(transaction)

        # Update portfolio cash balance
        portfolio.cash_balance -= required_capital

        db.session.commit()

        return jsonify({
            'status': 'success',
            'message': 'Position opened successfully',
            'data': {
                'position_id': position.id
            }
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
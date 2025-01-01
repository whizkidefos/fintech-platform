from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required, current_user
from app.models.portfolio import Portfolio
from app.models.position import Position
from app.models.order import Order
from app.models.signal import TradingSignal
from app.utils.order_executor import OrderExecutor
from app.utils.signal_generator import SignalGenerator
from app.utils.market_data import MarketDataFetcher
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd

api = Blueprint('api', __name__)
market_data = MarketDataFetcher()

@api.route('/market-data/<symbol>/ticker', methods=['GET'])
@login_required
def get_ticker(symbol):
    try:
        ticker = market_data.get_ticker(symbol)
        return jsonify(ticker)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@api.route('/market-data/<symbol>/candles', methods=['GET'])
@login_required
def get_candles(symbol):
    timeframe = request.args.get('timeframe', '15m')
    limit = int(request.args.get('limit', 100))
    
    try:
        candles = market_data.get_candles(symbol, timeframe, limit)
        return jsonify(candles)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@api.route('/portfolio/summary', methods=['GET'])
@login_required
def get_portfolio_summary():
    portfolio = Portfolio.query.filter_by(user_id=current_user.id).first()
    if not portfolio:
        return jsonify({'error': 'Portfolio not found'}), 404

    return jsonify({
        'total_value': float(portfolio.total_value),
        'cash_balance': float(portfolio.cash_balance),
        'daily_pnl': float(portfolio.daily_pnl),
        'daily_change': float(portfolio.daily_change),
        'open_positions': len(portfolio.positions)
    })

@api.route('/portfolio/positions', methods=['GET'])
@login_required
def get_positions():
    portfolio = Portfolio.query.filter_by(user_id=current_user.id).first()
    if not portfolio:
        return jsonify({'error': 'Portfolio not found'}), 404

    positions = []
    for position in portfolio.positions:
        current_price = market_data.get_ticker(position.symbol)['last']
        pnl = position.calculate_pnl(Decimal(str(current_price)))
        
        positions.append({
            'id': position.id,
            'symbol': position.symbol,
            'side': position.side,
            'size': float(position.size),
            'entry_price': float(position.entry_price),
            'current_price': float(current_price),
            'pnl': float(pnl),
            'pnl_percent': float(position.calculate_pnl_percent(Decimal(str(current_price))))
        })

    return jsonify(positions)

@api.route('/orders', methods=['POST'])
@login_required
def create_order():
    data = request.json
    portfolio = Portfolio.query.filter_by(user_id=current_user.id).first()
    
    if not portfolio:
        return jsonify({'error': 'Portfolio not found'}), 404

    try:
        order = Order(
            portfolio_id=portfolio.id,
            symbol=data['symbol'],
            side=data['side'],
            type=data['type'],
            size=Decimal(str(data['size'])),
            status='PENDING'
        )

        if 'price' in data:
            order.price = Decimal(str(data['price']))
        if 'stopLoss' in data:
            order.stop_loss = Decimal(str(data['stopLoss']))
        if 'takeProfit' in data:
            order.take_profit = Decimal(str(data['takeProfit']))

        executor = OrderExecutor(current_app.db.session, market_data)
        result = executor.execute_order(portfolio, order)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@api.route('/positions/<int:position_id>/close', methods=['POST'])
@login_required
def close_position(position_id):
    portfolio = Portfolio.query.filter_by(user_id=current_user.id).first()
    position = Position.query.get(position_id)

    if not portfolio or not position or position.portfolio_id != portfolio.id:
        return jsonify({'error': 'Position not found'}), 404

    try:
        executor = OrderExecutor(current_app.db.session, market_data)
        result = executor.close_position(position)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@api.route('/signals', methods=['GET'])
@login_required
def get_signals():
    symbol = request.args.get('symbol')
    limit = int(request.args.get('limit', 10))

    query = TradingSignal.query.filter_by(active=True)
    if symbol:
        query = query.filter_by(symbol=symbol)
    
    signals = query.order_by(TradingSignal.timestamp.desc()).limit(limit).all()
    
    return jsonify([{
        'id': signal.id,
        'symbol': signal.symbol,
        'type': signal.type,
        'price': float(signal.price),
        'strength': float(signal.strength),
        'timestamp': signal.timestamp.isoformat()
    } for signal in signals])

@api.route('/signals/<int:signal_id>/execute', methods=['POST'])
@login_required
def execute_signal(signal_id):
    portfolio = Portfolio.query.filter_by(user_id=current_user.id).first()
    signal = TradingSignal.query.get(signal_id)

    if not portfolio or not signal:
        return jsonify({'error': 'Signal not found'}), 404

    try:
        executor = OrderExecutor(current_app.db.session, market_data)
        order = Order(
            portfolio_id=portfolio.id,
            symbol=signal.symbol,
            side=signal.type,
            type='MARKET',
            size=signal.recommended_size,
            status='PENDING'
        )
        
        result = executor.execute_order(portfolio, order)
        signal.active = False
        current_app.db.session.commit()
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required, current_user
from app.models import Portfolio, Position, TradingSignal, Asset
from app.utils.order_executor import OrderExecutor
from app.utils.signal_generator import SignalGenerator
from app.utils.market_data import MarketDataFetcher
from datetime import datetime, timedelta
from decimal import Decimal
from app import db

# Create blueprint with url_prefix
bp = Blueprint('api', __name__, url_prefix='/api')
market_data = MarketDataFetcher()
order_executor = OrderExecutor()
signal_generator = SignalGenerator()

@bp.route('/ticker/<symbol>', methods=['GET'])
@login_required
def get_ticker(symbol):
    try:
        ticker_data = market_data.get_ticker(symbol)
        return jsonify(ticker_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@bp.route('/candles/<symbol>', methods=['GET'])
@login_required
def get_candles(symbol):
    timeframe = request.args.get('timeframe', '1h')
    limit = request.args.get('limit', 100, type=int)
    try:
        candles = market_data.get_candles(symbol, timeframe, limit)
        return jsonify(candles), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@bp.route('/portfolio/summary', methods=['GET'])
@login_required
def get_portfolio_summary():
    try:
        portfolio = Portfolio.query.filter_by(user_id=current_user.id).first()
        if not portfolio:
            return jsonify({'error': 'Portfolio not found'}), 404
            
        summary = {
            'total_value': portfolio.total_value,
            'total_cost': portfolio.total_cost,
            'unrealized_pnl': portfolio.unrealized_pnl,
            'risk_profile': portfolio.risk_profile
        }
        return jsonify(summary), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@bp.route('/positions', methods=['GET'])
@login_required
def get_positions():
    try:
        portfolio = Portfolio.query.filter_by(user_id=current_user.id).first()
        if not portfolio:
            return jsonify({'error': 'Portfolio not found'}), 404
            
        positions = []
        for position in portfolio.positions:
            pos_data = {
                'id': position.id,
                'symbol': position.asset.symbol,
                'quantity': position.quantity,
                'cost_basis': position.cost_basis,
                'current_value': position.current_value,
                'unrealized_pnl': position.unrealized_pnl,
                'return_pct': position.return_pct,
                'entry_date': position.entry_date.isoformat(),
                'last_updated': position.last_updated.isoformat()
            }
            positions.append(pos_data)
            
        return jsonify(positions), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@bp.route('/orders', methods=['POST'])
@login_required
def create_order():
    try:
        data = request.get_json()
        portfolio = Portfolio.query.filter_by(user_id=current_user.id).first()
        if not portfolio:
            return jsonify({'error': 'Portfolio not found'}), 404
            
        # Validate order parameters
        required_fields = ['symbol', 'side', 'quantity', 'order_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
                
        # Execute order
        order_result = order_executor.execute_order(
            portfolio_id=portfolio.id,
            symbol=data['symbol'],
            side=data['side'],
            quantity=Decimal(str(data['quantity'])),
            order_type=data['order_type'],
            price=data.get('price'),
            time_in_force=data.get('time_in_force', 'GTC')
        )
        
        return jsonify(order_result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@bp.route('/positions/<int:position_id>', methods=['DELETE'])
@login_required
def close_position(position_id):
    try:
        position = Position.query.get(position_id)
        if not position:
            return jsonify({'error': 'Position not found'}), 404
            
        if position.portfolio.user_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
            
        result = order_executor.close_position(position_id)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@bp.route('/signals', methods=['GET'])
@login_required
def get_signals():
    try:
        portfolio = Portfolio.query.filter_by(user_id=current_user.id).first()
        if not portfolio:
            return jsonify({'error': 'Portfolio not found'}), 404
            
        # Get active signals
        signals = TradingSignal.query.filter_by(
            portfolio_id=portfolio.id,
            status='active',
            executed=False
        ).all()
        
        signals_data = []
        for signal in signals:
            signal_data = {
                'id': signal.id,
                'symbol': signal.asset.symbol,
                'signal_type': signal.signal_type,
                'direction': signal.direction,
                'strength': signal.strength,
                'timestamp': signal.timestamp.isoformat(),
                'strategy': signal.strategy,
                'confidence': signal.confidence
            }
            signals_data.append(signal_data)
            
        return jsonify(signals_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@bp.route('/signals/<int:signal_id>/execute', methods=['POST'])
@login_required
def execute_signal(signal_id):
    try:
        signal = TradingSignal.query.get(signal_id)
        if not signal:
            return jsonify({'error': 'Signal not found'}), 404
            
        portfolio = Portfolio.query.filter_by(user_id=current_user.id).first()
        if not portfolio or signal.portfolio_id != portfolio.id:
            return jsonify({'error': 'Unauthorized'}), 403
            
        if signal.executed:
            return jsonify({'error': 'Signal already executed'}), 400
            
        # Execute the trading signal
        result = order_executor.execute_signal(signal_id)
        
        # Update signal status
        signal.executed = True
        signal.status = 'executed'
        db.session.commit()
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
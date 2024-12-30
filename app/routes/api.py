from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from app.utils.market_data import MarketDataFetcher
from app.utils.trading_signals import SignalGenerator
from app.models.asset import Asset, Portfolio, PortfolioAsset
from app.models.signal import TradingSignal
from app import db
from datetime import datetime

bp = Blueprint('api', __name__, url_prefix='/api')

market_data = MarketDataFetcher()

@bp.route('/market-data/<symbol>')
@login_required
def get_market_data(symbol):
    timeframe = request.args.get('timeframe', '1h')
    limit = int(request.args.get('limit', 100))
    
    data = market_data.fetch_ohlcv(symbol, timeframe, limit)
    if data is not None:
        return jsonify({
            'status': 'success',
            'data': data.to_dict(orient='records')
        })
    return jsonify({'status': 'error', 'message': 'Failed to fetch market data'})

@bp.route('/signals/<symbol>')
@login_required
def get_signals(symbol):
    timeframe = request.args.get('timeframe', '1h')
    limit = int(request.args.get('limit', 100))
    
    # Fetch market data
    data = market_data.fetch_ohlcv(symbol, timeframe, limit)
    if data is None:
        return jsonify({'status': 'error', 'message': 'Failed to fetch market data'})
    
    # Generate signals
    signal_generator = SignalGenerator(data)
    signals = signal_generator.generate_signals()
    
    # Store signals in database
    for signal in signals:
        asset = Asset.query.filter_by(symbol=symbol).first()
        if asset:
            db_signal = TradingSignal(
                asset_id=asset.id,
                signal_type=signal['type'],
                strength=signal['strength'],
                indicator_values=signal['indicators'],
                created_at=signal['timestamp']
            )
            db.session.add(db_signal)
    
    db.session.commit()
    
    return jsonify({
        'status': 'success',
        'data': signals
    })

@bp.route('/portfolio')
@login_required
def get_portfolio():
    portfolios = Portfolio.query.filter_by(user_id=current_user.id).all()
    portfolio_data = []
    
    for portfolio in portfolios:
        assets = []
        total_value = 0
        
        for position in portfolio.assets:
            asset = position.asset
            current_price = market_data.fetch_ticker(asset.symbol)['last_price']
            value = position.quantity * current_price
            total_value += value
            
            assets.append({
                'symbol': asset.symbol,
                'quantity': position.quantity,
                'current_price': current_price,
                'value': value,
                'avg_buy_price': position.average_buy_price,
                'profit_loss': value - (position.quantity * position.average_buy_price)
            })
        
        portfolio_data.append({
            'id': portfolio.id,
            'name': portfolio.name,
            'total_value': total_value,
            'assets': assets
        })
    
    return jsonify({
        'status': 'success',
        'data': portfolio_data
    })

@bp.route('/watchlist')
@login_required
def get_watchlist():
    # Implement watchlist functionality
    assets = Asset.query.all()  # Replace with proper watchlist query
    watchlist_data = []
    
    for asset in assets:
        ticker = market_data.fetch_ticker(asset.symbol)
        if ticker:
            watchlist_data.append({
                'symbol': asset.symbol,
                'name': asset.name,
                'current_price': ticker['last_price'],
                'volume': ticker['volume'],
                'bid': ticker['bid'],
                'ask': ticker['ask']
            })
    
    return jsonify({
        'status': 'success',
        'data': watchlist_data
    })

@bp.route('/execute-trade', methods=['POST'])
@login_required
def execute_trade():
    data = request.get_json()
    
    required_fields = ['symbol', 'type', 'quantity', 'portfolio_id']
    if not all(field in data for field in required_fields):
        return jsonify({'status': 'error', 'message': 'Missing required fields'})
    
    try:
        # Get current market price
        ticker = market_data.fetch_ticker(data['symbol'])
        if not ticker:
            return jsonify({'status': 'error', 'message': 'Failed to fetch market price'})
        
        # Get or create asset
        asset = Asset.query.filter_by(symbol=data['symbol']).first()
        if not asset:
            asset = Asset(symbol=data['symbol'])
            db.session.add(asset)
        
        # Create transaction
        transaction = Transaction(
            user_id=current_user.id,
            asset_id=asset.id,
            transaction_type=data['type'],
            quantity=data['quantity'],
            price=ticker['last_price']
        )
        db.session.add(transaction)
        
        # Update portfolio position
        portfolio = Portfolio.query.get(data['portfolio_id'])
        if not portfolio or portfolio.user_id != current_user.id:
            return jsonify({'status': 'error', 'message': 'Invalid portfolio'})
        
        position = PortfolioAsset.query.filter_by(
            portfolio_id=portfolio.id,
            asset_id=asset.id
        ).first()
        
        if position:
            if data['type'] == 'buy':
                new_quantity = position.quantity + data['quantity']
                position.average_buy_price = ((position.quantity * position.average_buy_price) + 
                                           (data['quantity'] * ticker['last_price'])) / new_quantity
                position.quantity = new_quantity
            else:  # sell
                if position.quantity < data['quantity']:
                    return jsonify({'status': 'error', 'message': 'Insufficient balance'})
                position.quantity -= data['quantity']
        else:
            if data['type'] == 'buy':
                position = PortfolioAsset(
                    portfolio_id=portfolio.id,
                    asset_id=asset.id,
                    quantity=data['quantity'],
                    average_buy_price=ticker['last_price']
                )
                db.session.add(position)
            else:
                return jsonify({'status': 'error', 'message': 'No position to sell'})
        
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': f'{data["type"].capitalize()} order executed successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)})
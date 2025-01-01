from flask import Blueprint, render_template
from flask_login import login_required, current_user
from app.utils.market_data import MarketDataFetcher
from app.models import Portfolio, Position, TradingSignal
from datetime import datetime, timedelta

bp = Blueprint('dashboard', __name__)
market_data = MarketDataFetcher()

@bp.route('/')
@bp.route('/dashboard')
@login_required
def index():
    portfolio = Portfolio.query.filter_by(user_id=current_user.id).first()
    
    # Get portfolio summary
    portfolio_summary = {
        'total_value': portfolio.total_value if portfolio else 0,
        'total_cost': portfolio.total_cost if portfolio else 0,
        'unrealized_pnl': portfolio.unrealized_pnl if portfolio else 0,
        'day_change': portfolio.day_change if portfolio else 0,
        'day_change_pct': portfolio.day_change_pct if portfolio else 0
    }
    
    # Get positions
    positions = []
    if portfolio:
        for position in portfolio.positions:
            pos_data = {
                'symbol': position.asset.symbol,
                'quantity': position.quantity,
                'cost_basis': position.cost_basis,
                'current_value': position.current_value,
                'unrealized_pnl': position.unrealized_pnl,
                'return_pct': position.return_pct,
                'day_change': position.day_change,
                'day_change_pct': position.day_change_pct
            }
            positions.append(pos_data)
    
    # Get recent signals
    recent_signals = []
    if portfolio:
        signals = TradingSignal.query.filter_by(
            portfolio_id=portfolio.id,
            status='active'
        ).order_by(TradingSignal.timestamp.desc()).limit(5).all()
        
        for signal in signals:
            signal_data = {
                'symbol': signal.asset.symbol,
                'signal_type': signal.signal_type,
                'direction': signal.direction,
                'strength': signal.strength,
                'timestamp': signal.timestamp,
                'strategy': signal.strategy
            }
            recent_signals.append(signal_data)
    
    # Get market overview
    market_indices = ['^GSPC', '^DJI', '^IXIC']  # S&P 500, Dow, Nasdaq
    market_overview = []
    
    for symbol in market_indices:
        try:
            ticker_data = market_data.get_ticker(symbol)
            market_overview.append({
                'symbol': symbol,
                'price': ticker_data['price'],
                'change': ticker_data['change'],
                'change_pct': ticker_data['change_pct']
            })
        except Exception as e:
            continue
    
    return render_template('dashboard/index.html',
                         portfolio_summary=portfolio_summary,
                         positions=positions,
                         recent_signals=recent_signals,
                         market_overview=market_overview)

@bp.route('/portfolio')
@login_required
def portfolio():
    portfolio = Portfolio.query.filter_by(user_id=current_user.id).first()
    return render_template('dashboard/portfolio.html', portfolio=portfolio)

@bp.route('/signals')
@login_required
def signals():
    portfolio = Portfolio.query.filter_by(user_id=current_user.id).first()
    if portfolio:
        signals = TradingSignal.query.filter_by(
            portfolio_id=portfolio.id
        ).order_by(TradingSignal.timestamp.desc()).all()
    else:
        signals = []
    return render_template('dashboard/signals.html', signals=signals)

@bp.route('/market')
@login_required
def market():
    # Get market overview data
    market_data_items = market_data.get_market_overview()
    return render_template('dashboard/market.html',
                         market_data=market_data_items)
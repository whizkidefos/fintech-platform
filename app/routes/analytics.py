from flask import Blueprint, jsonify, request, send_file
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from io import StringIO
from app.utils.analytics import PortfolioAnalytics, TradeAnalytics, RiskAnalytics
from app.models.portfolio import Portfolio
from app.models.trade import Trade
from app.utils.auth import login_required

analytics_bp = Blueprint('analytics', __name__)

@analytics_bp.route('/api/analytics', methods=['GET'])
@login_required
def get_analytics():
    timeframe = request.args.get('timeframe', '1M')
    end_date = datetime.now()
    
    # Calculate start date based on timeframe
    if timeframe == '1M':
        start_date = end_date - timedelta(days=30)
    elif timeframe == '3M':
        start_date = end_date - timedelta(days=90)
    elif timeframe == '6M':
        start_date = end_date - timedelta(days=180)
    elif timeframe == '1Y':
        start_date = end_date - timedelta(days=365)
    else:  # ALL
        start_date = None
    
    # Get portfolio data
    portfolio = Portfolio.get_by_user_id(current_user.id)
    trades = Trade.get_by_date_range(current_user.id, start_date, end_date)
    
    # Initialize analytics classes
    portfolio_analytics = PortfolioAnalytics()
    trade_analytics = TradeAnalytics()
    risk_analytics = RiskAnalytics()
    
    # Calculate portfolio metrics
    portfolio_values = portfolio.get_historical_values(start_date, end_date)
    returns = portfolio_analytics.calculate_returns(portfolio_values)
    metrics = portfolio_analytics.calculate_metrics(returns)
    
    # Calculate risk metrics
    var = portfolio_analytics.calculate_var(returns)
    cvar = portfolio_analytics.calculate_cvar(returns)
    
    # Get market data for beta calculation
    market_returns = get_market_returns(start_date, end_date)  # Implement this function
    beta = portfolio_analytics.calculate_beta(returns, market_returns)
    alpha = portfolio_analytics.calculate_alpha(returns, market_returns)
    
    # Analyze trades
    trade_analysis = trade_analytics.analyze_trades(trades)
    
    # Calculate return distribution
    returns_distribution = calculate_returns_distribution(returns)
    
    response_data = {
        'equity_curve': {
            'dates': portfolio_values.index.tolist(),
            'values': portfolio_values.values.tolist()
        },
        'metrics': {
            'total_return': float(metrics.total_return),
            'sharpe_ratio': float(metrics.sharpe_ratio),
            'sortino_ratio': float(metrics.sortino_ratio),
            'max_drawdown': float(metrics.max_drawdown),
            'win_rate': float(metrics.win_rate),
            'profit_factor': float(metrics.profit_factor),
            'avg_trade': float(metrics.avg_trade),
            'trades_count': metrics.trades_count
        },
        'risk_metrics': {
            'var': float(var),
            'cvar': float(cvar),
            'beta': float(beta),
            'alpha': float(alpha)
        },
        'returns_distribution': returns_distribution,
        'performance': {
            'hourly': format_performance_data(trade_analysis['hourly_performance']),
            'daily': format_performance_data(trade_analysis['daily_performance']),
            'symbol': format_performance_data(trade_analysis['symbol_performance']),
            'strategy': format_performance_data(trade_analysis['strategy_performance'])
        }
    }
    
    return jsonify(response_data)

@analytics_bp.route('/api/analytics/export', methods=['GET'])
@login_required
def export_analytics():
    timeframe = request.args.get('timeframe', '1M')
    end_date = datetime.now()
    
    # Calculate start date based on timeframe (same as above)
    if timeframe == '1M':
        start_date = end_date - timedelta(days=30)
    elif timeframe == '3M':
        start_date = end_date - timedelta(days=90)
    elif timeframe == '6M':
        start_date = end_date - timedelta(days=180)
    elif timeframe == '1Y':
        start_date = end_date - timedelta(days=365)
    else:  # ALL
        start_date = None
    
    # Get trade data
    trades = Trade.get_by_date_range(current_user.id, start_date, end_date)
    
    # Convert to DataFrame
    df = pd.DataFrame(trades)
    
    # Add additional analysis columns
    df['cumulative_pnl'] = df['pnl'].cumsum()
    df['win'] = df['pnl'] > 0
    df['holding_time'] = df['exit_time'] - df['entry_time']
    
    # Create CSV
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    
    # Create response
    output = csv_buffer.getvalue()
    csv_buffer.close()
    
    return send_file(
        StringIO(output),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'trade_analysis_{timeframe}_{datetime.now().strftime("%Y%m%d")}.csv'
    )

def calculate_returns_distribution(returns):
    """Calculate return distribution for histogram"""
    hist, bins = np.histogram(returns, bins=50)
    return {
        'frequencies': hist.tolist(),
        'bins': [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
    }

def format_performance_data(performance_dict):
    """Format performance data for charts"""
    return {
        'labels': list(performance_dict.keys()),
        'values': [float(val['sum']) for val in performance_dict.values()]
    }

def get_market_returns(start_date, end_date):
    """Get market returns for beta calculation
    Implement this based on your market data source
    """
    # Example implementation using a market index
    market_data = get_market_index_data(start_date, end_date)  # Implement this
    return pd.Series(market_data['returns'], index=market_data['dates'])

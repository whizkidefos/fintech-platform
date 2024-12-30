from flask import Blueprint, render_template
from flask_login import login_required, current_user
from app.models.asset import Portfolio, Asset
from app.models.signal import TradingSignal

bp = Blueprint('dashboard', __name__)

@bp.route('/')
@bp.route('/dashboard')
@login_required
def index():
    portfolios = Portfolio.query.filter_by(user_id=current_user.id).all()
    recent_signals = TradingSignal.query.order_by(TradingSignal.created_at.desc()).limit(5).all()
    watchlist = Asset.query.limit(10).all()  # Implement proper watchlist later
    
    return render_template('dashboard/index.html',
                         portfolios=portfolios,
                         recent_signals=recent_signals,
                         watchlist=watchlist)
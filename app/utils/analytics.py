import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from decimal import Decimal

@dataclass
class PerformanceMetrics:
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade: float
    avg_win: float
    avg_loss: float
    risk_reward_ratio: float
    expectancy: float
    trades_count: int
    profitable_trades: int
    unprofitable_trades: int

class PortfolioAnalytics:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate

    def calculate_returns(self, portfolio_values: pd.Series) -> pd.Series:
        """Calculate periodic returns"""
        return portfolio_values.pct_change().dropna()

    def calculate_metrics(self, returns: pd.Series) -> PerformanceMetrics:
        """Calculate all performance metrics"""
        total_return = (returns + 1).prod() - 1
        volatility = returns.std() * np.sqrt(252)  # Annualized
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        
        # Sharpe and Sortino Ratios
        excess_returns = returns - (self.risk_free_rate / 252)
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / volatility
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_volatility
        
        # Maximum Drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdowns = cum_returns / running_max - 1
        max_drawdown = drawdowns.min()
        
        # Trading Statistics
        trades = returns[returns != 0]
        wins = trades[trades > 0]
        losses = trades[trades < 0]
        
        win_rate = len(wins) / len(trades) if len(trades) > 0 else 0
        profit_factor = abs(wins.sum() / losses.sum()) if len(losses) > 0 else float('inf')
        avg_trade = trades.mean()
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        return PerformanceMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade=avg_trade,
            avg_win=avg_win,
            avg_loss=avg_loss,
            risk_reward_ratio=risk_reward,
            expectancy=expectancy,
            trades_count=len(trades),
            profitable_trades=len(wins),
            unprofitable_trades=len(losses)
        )

    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100)

    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self.calculate_var(returns, confidence)
        return returns[returns <= var].mean()

    def calculate_beta(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate portfolio beta"""
        covariance = returns.cov(market_returns)
        market_variance = market_returns.var()
        return covariance / market_variance if market_variance != 0 else 1

    def calculate_alpha(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate portfolio alpha"""
        beta = self.calculate_beta(returns, market_returns)
        return (returns.mean() - self.risk_free_rate) - (beta * (market_returns.mean() - self.risk_free_rate))

class TradeAnalytics:
    @staticmethod
    def analyze_trades(trades: List[Dict]) -> Dict:
        """Analyze trading performance by various factors"""
        df = pd.DataFrame(trades)
        
        # Time-based analysis
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        hourly_performance = df.groupby('hour')['pnl'].agg(['mean', 'count', 'sum'])
        daily_performance = df.groupby('day_of_week')['pnl'].agg(['mean', 'count', 'sum'])
        
        # Symbol analysis
        symbol_performance = df.groupby('symbol')['pnl'].agg(['mean', 'count', 'sum'])
        
        # Strategy analysis
        strategy_performance = df.groupby('strategy')['pnl'].agg(['mean', 'count', 'sum'])
        
        return {
            'hourly_performance': hourly_performance.to_dict(),
            'daily_performance': daily_performance.to_dict(),
            'symbol_performance': symbol_performance.to_dict(),
            'strategy_performance': strategy_performance.to_dict()
        }

    @staticmethod
    def calculate_position_sizing(account_balance: Decimal, risk_per_trade: float,
                                entry_price: Decimal, stop_loss: Decimal) -> Decimal:
        """Calculate position size based on risk management rules"""
        risk_amount = Decimal(str(account_balance * Decimal(str(risk_per_trade))))
        price_difference = abs(entry_price - stop_loss)
        
        if price_difference == 0:
            return Decimal('0')
            
        position_size = risk_amount / price_difference
        return position_size

class RiskAnalytics:
    @staticmethod
    def calculate_portfolio_risk(positions: List[Dict], correlations: pd.DataFrame) -> Dict:
        """Calculate portfolio risk metrics"""
        position_weights = []
        position_returns = []
        
        for position in positions:
            weight = Decimal(str(position['size'] * position['current_price']))
            position_weights.append(weight)
            position_returns.append(position['returns'])
            
        total_value = sum(position_weights)
        weights = [w / total_value for w in position_weights]
        
        # Portfolio volatility
        portfolio_variance = 0
        for i, w1 in enumerate(weights):
            for j, w2 in enumerate(weights):
                if i != j:
                    portfolio_variance += (w1 * w2 * correlations.iloc[i, j] * 
                                        position_returns[i].std() * position_returns[j].std())
                else:
                    portfolio_variance += w1 * w1 * position_returns[i].var()
                    
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        return {
            'portfolio_volatility': float(portfolio_volatility),
            'position_contributions': [float(w * w * r.var()) for w, r in zip(weights, position_returns)]
        }

    @staticmethod
    def stress_test_portfolio(positions: List[Dict], scenarios: List[Dict]) -> Dict:
        """Perform stress testing on portfolio"""
        results = {}
        
        for scenario in scenarios:
            total_pnl = Decimal('0')
            for position in positions:
                price_change = Decimal(str(scenario['price_changes'].get(position['symbol'], 0)))
                position_pnl = position['size'] * price_change
                total_pnl += position_pnl
                
            results[scenario['name']] = float(total_pnl)
            
        return results

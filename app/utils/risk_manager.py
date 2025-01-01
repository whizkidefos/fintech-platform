from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from app.models import Portfolio, Position, Asset
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    var: float  # Value at Risk
    es: float   # Expected Shortfall
    beta: float # Portfolio Beta
    sharpe: float # Sharpe Ratio
    volatility: float # Portfolio Volatility
    max_drawdown: float # Maximum Drawdown
    correlation_matrix: np.ndarray # Asset Correlations
    position_limits: Dict[str, float] # Position Size Limits
    leverage: float # Current Portfolio Leverage

class RiskManager:
    def __init__(self, confidence_level: float = 0.95, 
                 lookback_days: int = 252,
                 risk_free_rate: float = 0.02):
        self.confidence_level = confidence_level
        self.lookback_days = lookback_days
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)

    def calculate_portfolio_risk(self, portfolio: Portfolio) -> RiskMetrics:
        """Calculate comprehensive risk metrics for a portfolio"""
        try:
            # Get position data
            positions = portfolio.positions
            if not positions:
                return self._get_empty_risk_metrics()

            # Calculate returns and weights
            returns, weights = self._get_portfolio_data(positions)
            
            # Calculate risk metrics
            var = self._calculate_var(returns, weights)
            es = self._calculate_expected_shortfall(returns, weights)
            beta = self._calculate_portfolio_beta(returns)
            sharpe = self._calculate_sharpe_ratio(returns, weights)
            vol = self._calculate_portfolio_volatility(returns, weights)
            max_dd = self._calculate_max_drawdown(returns, weights)
            corr_matrix = self._calculate_correlation_matrix(returns)
            pos_limits = self._calculate_position_limits(positions)
            leverage = self._calculate_portfolio_leverage(positions)

            return RiskMetrics(
                var=var,
                es=es,
                beta=beta,
                sharpe=sharpe,
                volatility=vol,
                max_drawdown=max_dd,
                correlation_matrix=corr_matrix,
                position_limits=pos_limits,
                leverage=leverage
            )
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk: {str(e)}")
            return self._get_empty_risk_metrics()

    def check_order_risk(self, portfolio: Portfolio, 
                        symbol: str, 
                        quantity: float, 
                        side: str) -> Tuple[bool, str]:
        """Check if an order meets risk requirements"""
        try:
            # Get current position if exists
            current_position = next(
                (p for p in portfolio.positions if p.asset.symbol == symbol),
                None
            )

            # Calculate new position size
            new_quantity = quantity
            if current_position:
                if side == 'buy':
                    new_quantity += current_position.quantity
                else:
                    new_quantity = current_position.quantity - quantity

            # Get risk metrics
            metrics = self.calculate_portfolio_risk(portfolio)

            # Check position limits
            if not self._check_position_limits(new_quantity, symbol, metrics):
                return False, "Position size exceeds limits"

            # Check leverage
            if not self._check_leverage(portfolio, new_quantity, symbol):
                return False, "Order would exceed leverage limits"

            # Check concentration
            if not self._check_concentration(portfolio, new_quantity, symbol):
                return False, "Order would exceed concentration limits"

            # Check correlation
            if not self._check_correlation(portfolio, symbol, metrics):
                return False, "High correlation with existing positions"

            return True, "Order meets risk requirements"
        except Exception as e:
            self.logger.error(f"Error checking order risk: {str(e)}")
            return False, f"Risk check error: {str(e)}"

    def _get_empty_risk_metrics(self) -> RiskMetrics:
        """Return empty risk metrics for empty portfolio"""
        return RiskMetrics(
            var=0.0,
            es=0.0,
            beta=0.0,
            sharpe=0.0,
            volatility=0.0,
            max_drawdown=0.0,
            correlation_matrix=np.array([]),
            position_limits={},
            leverage=0.0
        )

    def _get_portfolio_data(self, positions: List[Position]) -> Tuple[np.ndarray, np.ndarray]:
        """Get historical returns and current weights for portfolio positions"""
        # Implementation would fetch historical price data and calculate returns
        # This is a simplified version
        returns = np.random.normal(0.0001, 0.02, (self.lookback_days, len(positions)))
        total_value = sum(p.current_value for p in positions)
        weights = np.array([p.current_value/total_value for p in positions])
        return returns, weights

    def _calculate_var(self, returns: np.ndarray, weights: np.ndarray) -> float:
        """Calculate Value at Risk using historical simulation"""
        portfolio_returns = np.dot(returns, weights)
        var = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        return float(-var)

    def _calculate_expected_shortfall(self, returns: np.ndarray, weights: np.ndarray) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        portfolio_returns = np.dot(returns, weights)
        var = self._calculate_var(returns, weights)
        es = -np.mean(portfolio_returns[portfolio_returns <= -var])
        return float(es)

    def _calculate_portfolio_beta(self, returns: np.ndarray) -> float:
        """Calculate portfolio beta relative to market"""
        # Implementation would use market index returns
        # This is a simplified version
        return float(np.random.normal(1, 0.2))

    def _calculate_sharpe_ratio(self, returns: np.ndarray, weights: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        portfolio_returns = np.dot(returns, weights)
        excess_returns = portfolio_returns - self.risk_free_rate/252
        sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(portfolio_returns)
        return float(sharpe)

    def _calculate_portfolio_volatility(self, returns: np.ndarray, weights: np.ndarray) -> float:
        """Calculate portfolio volatility"""
        portfolio_returns = np.dot(returns, weights)
        return float(np.std(portfolio_returns) * np.sqrt(252))

    def _calculate_max_drawdown(self, returns: np.ndarray, weights: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        portfolio_returns = np.dot(returns, weights)
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns / running_max - 1
        return float(-np.min(drawdowns))

    def _calculate_correlation_matrix(self, returns: np.ndarray) -> np.ndarray:
        """Calculate correlation matrix between assets"""
        return np.corrcoef(returns.T)

    def _calculate_position_limits(self, positions: List[Position]) -> Dict[str, float]:
        """Calculate position limits based on risk factors"""
        limits = {}
        for position in positions:
            # Implementation would consider various factors
            # This is a simplified version
            limits[position.asset.symbol] = position.current_value * 2
        return limits

    def _calculate_portfolio_leverage(self, positions: List[Position]) -> float:
        """Calculate portfolio leverage"""
        total_exposure = sum(abs(p.current_value) for p in positions)
        net_value = sum(p.current_value for p in positions)
        return total_exposure / net_value if net_value else 0

    def _check_position_limits(self, quantity: float, symbol: str, metrics: RiskMetrics) -> bool:
        """Check if position size is within limits"""
        if symbol in metrics.position_limits:
            return quantity <= metrics.position_limits[symbol]
        return True

    def _check_leverage(self, portfolio: Portfolio, quantity: float, symbol: str) -> bool:
        """Check if order would exceed leverage limits"""
        # Implementation would calculate new leverage
        # This is a simplified version
        return True

    def _check_concentration(self, portfolio: Portfolio, quantity: float, symbol: str) -> bool:
        """Check if order would exceed concentration limits"""
        # Implementation would calculate concentration metrics
        # This is a simplified version
        return True

    def _check_correlation(self, portfolio: Portfolio, symbol: str, metrics: RiskMetrics) -> bool:
        """Check correlation with existing positions"""
        # Implementation would check correlation thresholds
        # This is a simplified version
        return True

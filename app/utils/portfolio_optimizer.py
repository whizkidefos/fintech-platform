import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class OptimizationResult:
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    metadata: Dict

class PortfolioOptimizer:
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.mean_returns = returns.mean() * 252
        self.cov_matrix = returns.cov() * 252
        self.num_assets = len(returns.columns)

    def optimize_minimum_volatility(self, constraints: Optional[Dict] = None) -> OptimizationResult:
        """Find the minimum volatility portfolio"""
        constraints = constraints or {}
        
        # Define constraints
        bounds = self._get_bounds(constraints)
        constraints_list = self._get_constraints(constraints)
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/self.num_assets] * self.num_assets)
        
        # Optimize
        result = minimize(
            self._portfolio_volatility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        weights = result.x
        volatility = self._portfolio_volatility(weights)
        expected_return = self._portfolio_return(weights)
        sharpe = self._portfolio_sharpe(weights)
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            metadata={'optimization_success': result.success}
        )

    def optimize_maximum_sharpe(self, constraints: Optional[Dict] = None) -> OptimizationResult:
        """Find the maximum Sharpe ratio portfolio"""
        constraints = constraints or {}
        
        bounds = self._get_bounds(constraints)
        constraints_list = self._get_constraints(constraints)
        
        initial_weights = np.array([1/self.num_assets] * self.num_assets)
        
        result = minimize(
            lambda w: -self._portfolio_sharpe(w),
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        weights = result.x
        volatility = self._portfolio_volatility(weights)
        expected_return = self._portfolio_return(weights)
        sharpe = self._portfolio_sharpe(weights)
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            metadata={'optimization_success': result.success}
        )

    def optimize_efficient_return(self, target_return: float,
                                constraints: Optional[Dict] = None) -> OptimizationResult:
        """Find the minimum volatility portfolio for a target return"""
        constraints = constraints or {}
        
        bounds = self._get_bounds(constraints)
        constraints_list = self._get_constraints(constraints)
        
        # Add return constraint
        return_constraint = {
            'type': 'eq',
            'fun': lambda w: self._portfolio_return(w) - target_return
        }
        constraints_list.append(return_constraint)
        
        initial_weights = np.array([1/self.num_assets] * self.num_assets)
        
        result = minimize(
            self._portfolio_volatility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        weights = result.x
        volatility = self._portfolio_volatility(weights)
        expected_return = self._portfolio_return(weights)
        sharpe = self._portfolio_sharpe(weights)
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            metadata={'optimization_success': result.success}
        )

    def optimize_risk_parity(self, constraints: Optional[Dict] = None) -> OptimizationResult:
        """Find the risk parity portfolio"""
        constraints = constraints or {}
        
        bounds = self._get_bounds(constraints)
        constraints_list = self._get_constraints(constraints)
        
        initial_weights = np.array([1/self.num_assets] * self.num_assets)
        
        result = minimize(
            self._risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        weights = result.x
        volatility = self._portfolio_volatility(weights)
        expected_return = self._portfolio_return(weights)
        sharpe = self._portfolio_sharpe(weights)
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            metadata={
                'optimization_success': result.success,
                'risk_contributions': self._get_risk_contributions(weights)
            }
        )

    def optimize_maximum_diversification(self, constraints: Optional[Dict] = None) -> OptimizationResult:
        """Find the maximum diversification portfolio"""
        constraints = constraints or {}
        
        bounds = self._get_bounds(constraints)
        constraints_list = self._get_constraints(constraints)
        
        initial_weights = np.array([1/self.num_assets] * self.num_assets)
        
        result = minimize(
            lambda w: -self._diversification_ratio(w),
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        weights = result.x
        volatility = self._portfolio_volatility(weights)
        expected_return = self._portfolio_return(weights)
        sharpe = self._portfolio_sharpe(weights)
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            metadata={
                'optimization_success': result.success,
                'diversification_ratio': self._diversification_ratio(weights)
            }
        )

    def _portfolio_return(self, weights: np.ndarray) -> float:
        """Calculate portfolio return"""
        return np.sum(self.mean_returns * weights)

    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility"""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

    def _portfolio_sharpe(self, weights: np.ndarray) -> float:
        """Calculate portfolio Sharpe ratio"""
        ret = self._portfolio_return(weights)
        vol = self._portfolio_volatility(weights)
        return (ret - self.risk_free_rate) / vol if vol != 0 else -np.inf

    def _risk_parity_objective(self, weights: np.ndarray) -> float:
        """Objective function for risk parity optimization"""
        risk_contributions = self._get_risk_contributions(weights)
        target_risk = 1.0 / self.num_assets
        return np.sum((risk_contributions - target_risk) ** 2)

    def _diversification_ratio(self, weights: np.ndarray) -> float:
        """Calculate portfolio diversification ratio"""
        weighted_vols = np.sqrt(np.diag(self.cov_matrix)) * weights
        portfolio_vol = self._portfolio_volatility(weights)
        return np.sum(weighted_vols) / portfolio_vol if portfolio_vol != 0 else 0

    def _get_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """Calculate risk contributions for each asset"""
        portfolio_vol = self._portfolio_volatility(weights)
        marginal_risk = np.dot(self.cov_matrix, weights)
        risk_contributions = weights * marginal_risk / portfolio_vol if portfolio_vol != 0 else np.zeros_like(weights)
        return risk_contributions

    def _get_bounds(self, constraints: Dict) -> List[Tuple[float, float]]:
        """Get bounds for optimization"""
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)
        return [(min_weight, max_weight) for _ in range(self.num_assets)]

    def _get_constraints(self, constraints: Dict) -> List[Dict]:
        """Get constraints for optimization"""
        base_constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        # Add sector constraints if specified
        if 'sector_constraints' in constraints:
            for sector, (min_weight, max_weight) in constraints['sector_constraints'].items():
                sector_assets = constraints['sector_mapping'][sector]
                base_constraints.extend([
                    {'type': 'ineq', 'fun': lambda x, s=sector_assets: np.sum(x[s]) - min_weight},
                    {'type': 'ineq', 'fun': lambda x, s=sector_assets: max_weight - np.sum(x[s])}
                ])
        
        return base_constraints

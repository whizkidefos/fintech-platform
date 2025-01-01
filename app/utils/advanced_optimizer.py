import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cvxopt
from sklearn.covariance import LedoitWolf
from statsmodels.regression.linear_model import OLS

@dataclass
class BlackLittermanResult:
    weights: np.ndarray
    expected_returns: np.ndarray
    covariance: np.ndarray
    omega: np.ndarray
    posterior_returns: np.ndarray
    posterior_covariance: np.ndarray

class AdvancedPortfolioOptimizer:
    def __init__(self, returns: pd.DataFrame, market_caps: Optional[np.ndarray] = None,
                 risk_free_rate: float = 0.02):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.market_caps = market_caps
        self.ledoit_wolf = LedoitWolf()
        self.ledoit_wolf.fit(returns)
        
        # Calculate robust covariance matrix
        self.robust_cov_matrix = self.ledoit_wolf.covariance_
        self.mean_returns = returns.mean() * 252

    def black_litterman_optimize(self, views: List[Dict], 
                               confidences: List[float],
                               tau: float = 0.025) -> BlackLittermanResult:
        """
        Implement Black-Litterman portfolio optimization
        
        Args:
            views: List of dictionaries containing views on assets
            confidences: Confidence levels for each view
            tau: Parameter indicating uncertainty in prior
        """
        # Calculate market equilibrium returns
        if self.market_caps is None:
            market_weights = np.ones(len(self.returns.columns)) / len(self.returns.columns)
        else:
            market_weights = self.market_caps / np.sum(self.market_caps)
        
        # Prior expected returns (market equilibrium)
        pi = self.risk_free_rate + market_weights @ self.robust_cov_matrix @ market_weights
        
        # Create views matrix P and views vector Q
        n_assets = len(self.returns.columns)
        n_views = len(views)
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        
        for i, view in enumerate(views):
            for asset, weight in view['weights'].items():
                asset_idx = self.returns.columns.get_loc(asset)
                P[i, asset_idx] = weight
            Q[i] = view['return']
        
        # Create diagonal uncertainty matrix for views
        omega = np.diag([1/conf for conf in confidences])
        
        # Calculate posterior parameters
        sigma_prior = tau * self.robust_cov_matrix
        sigma_post = np.linalg.inv(
            np.linalg.inv(sigma_prior) + 
            P.T @ np.linalg.inv(omega) @ P
        )
        mu_post = sigma_post @ (
            np.linalg.inv(sigma_prior) @ pi + 
            P.T @ np.linalg.inv(omega) @ Q
        )
        
        # Optimize with posterior parameters
        def objective(w):
            return -(w @ mu_post - 0.5 * self.risk_free_rate * w @ sigma_post @ w)
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x}
        ]
        
        result = minimize(
            objective,
            market_weights,
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, 1) for _ in range(n_assets)]
        )
        
        return BlackLittermanResult(
            weights=result.x,
            expected_returns=pi,
            covariance=self.robust_cov_matrix,
            omega=omega,
            posterior_returns=mu_post,
            posterior_covariance=sigma_post
        )

    def factor_optimization(self, factor_exposures: pd.DataFrame,
                          factor_returns: pd.DataFrame,
                          target_exposures: Dict[str, float]) -> np.ndarray:
        """
        Optimize portfolio based on factor exposures
        
        Args:
            factor_exposures: DataFrame of asset exposures to factors
            factor_returns: DataFrame of factor returns
            target_exposures: Dictionary of target factor exposures
        """
        n_assets = len(self.returns.columns)
        
        # Calculate factor covariance matrix
        factor_cov = factor_returns.cov().values
        
        # Calculate specific returns
        factor_model = OLS(self.returns, factor_exposures).fit()
        specific_returns = factor_model.resid
        specific_risk = np.diag(specific_returns.var())
        
        # Calculate total covariance matrix
        total_cov = (factor_exposures.values @ factor_cov @ factor_exposures.values.T +
                    specific_risk)
        
        def objective(w):
            # Minimize tracking error to target factor exposures
            portfolio_exposures = w @ factor_exposures.values
            tracking_error = np.sum((portfolio_exposures - 
                                   np.array(list(target_exposures.values()))) ** 2)
            return tracking_error + 0.5 * w @ total_cov @ w
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x}
        ]
        
        result = minimize(
            objective,
            np.ones(n_assets) / n_assets,
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, 1) for _ in range(n_assets)]
        )
        
        return result.x

    def regime_based_optimization(self, regime_probabilities: np.ndarray,
                                regime_returns: List[pd.DataFrame],
                                regime_covs: List[np.ndarray]) -> np.ndarray:
        """
        Optimize portfolio considering different market regimes
        
        Args:
            regime_probabilities: Probabilities of each regime
            regime_returns: List of returns for each regime
            regime_covs: List of covariance matrices for each regime
        """
        n_assets = len(self.returns.columns)
        n_regimes = len(regime_probabilities)
        
        # Calculate regime-weighted expected returns and covariance
        expected_returns = np.zeros(n_assets)
        expected_cov = np.zeros((n_assets, n_assets))
        
        for i in range(n_regimes):
            expected_returns += regime_probabilities[i] * regime_returns[i].mean().values
            expected_cov += regime_probabilities[i] * regime_covs[i]
        
        def objective(w):
            return -(w @ expected_returns - 0.5 * self.risk_free_rate * w @ expected_cov @ w)
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x}
        ]
        
        result = minimize(
            objective,
            np.ones(n_assets) / n_assets,
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, 1) for _ in range(n_assets)]
        )
        
        return result.x

    def robust_optimization(self, uncertainty_sets: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """
        Implement robust optimization considering parameter uncertainty
        
        Args:
            uncertainty_sets: Dictionary of uncertainty bounds for parameters
        """
        n_assets = len(self.returns.columns)
        
        # Calculate robust estimates using elliptical uncertainty sets
        def worst_case_return(w, uncertainty_set):
            mean_return = w @ self.mean_returns
            uncertainty = np.sqrt(w @ self.robust_cov_matrix @ w) * uncertainty_set[1]
            return mean_return - uncertainty
        
        def objective(w):
            return -worst_case_return(w, uncertainty_sets['returns'])
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x}
        ]
        
        result = minimize(
            objective,
            np.ones(n_assets) / n_assets,
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, 1) for _ in range(n_assets)]
        )
        
        return result.x

    def hierarchical_risk_parity(self) -> np.ndarray:
        """
        Implement Hierarchical Risk Parity portfolio optimization
        """
        # Calculate correlation matrix
        corr = self.returns.corr().values
        
        # Calculate distance matrix
        dist = np.sqrt(2 * (1 - corr))
        
        # Clustering using single linkage
        links = self._get_clusters(dist)
        sorted_idx = self._quasi_diag(links)
        weights = self._get_hrp_weights(self.returns.iloc[:, sorted_idx])
        
        # Reorder weights to match original order
        reordered_weights = np.zeros(len(weights))
        reordered_weights[sorted_idx] = weights
        
        return reordered_weights

    def _get_clusters(self, dist: np.ndarray) -> List:
        """Helper function for hierarchical clustering"""
        n = len(dist)
        clusters = [(i,) for i in range(n)]
        while len(clusters) > 1:
            min_dist = float('inf')
            min_i, min_j = None, None
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    d = self._cluster_distance(clusters[i], clusters[j], dist)
                    if d < min_dist:
                        min_dist = d
                        min_i, min_j = i, j
            
            new_cluster = clusters[min_i] + clusters[min_j]
            clusters = [c for k, c in enumerate(clusters) if k not in [min_i, min_j]]
            clusters.append(new_cluster)
        
        return clusters[0]

    def _cluster_distance(self, cluster1: Tuple, cluster2: Tuple,
                         dist: np.ndarray) -> float:
        """Calculate distance between clusters"""
        return min(dist[i, j] for i in cluster1 for j in cluster2)

    def _quasi_diag(self, links: Tuple) -> List[int]:
        """Re-order elements for hierarchical tree"""
        return list(links)

    def _get_hrp_weights(self, cov: pd.DataFrame) -> np.ndarray:
        """Calculate HRP weights"""
        w = pd.Series(1, index=cov.columns)
        cluster_items = [cov.index.tolist()]
        
        while len(cluster_items) > 0:
            cluster_items = [
                i[j:k] for i in cluster_items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]
            
            for items in cluster_items:
                var = cov[items].var().values
                w[items] *= var.sum() / var / len(items)
        
        return w / w.sum()

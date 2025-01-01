from typing import Dict, List, Any
import pandas as pd
import numpy as np
from scipy import stats

class PortfolioAnalytics:
    def __init__(self, portfolio_data: pd.DataFrame):
        """
        Initialize with portfolio data
        portfolio_data should contain:
        - timestamp
        - asset_values (dictionary of asset:value)
        - cash_value
        """
        self.data = portfolio_data
        self.risk_free_rate = 0.02  # Adjustable risk-free rate

    def calculate_returns(self) -> Dict[str, float]:
        """Calculate basic return metrics"""
        total_values = self.data['asset_values'].apply(lambda x: sum(x.values())) + self.data['cash_value']
        returns = total_values.pct_change()
        
        return {
            'total_return': (total_values.iloc[-1] / total_values.iloc[0] - 1) * 100,
            'daily_returns': returns,
            'average_daily_return': returns.mean() * 100,
            'volatility': returns.std() * np.sqrt(252) * 100,  # Annualized volatility
        }

    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk-related metrics"""
        returns = self.calculate_returns()['daily_returns']
        
        # Sharpe Ratio
        excess_returns = returns - self.risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std()
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
        
        # Maximum Drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown * 100,
            'var_95': returns.quantile(0.05) * 100,  # 95% VaR
            'cvar_95': returns[returns <= returns.quantile(0.05)].mean() * 100  # 95% CVaR
        }

    def calculate_asset_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for individual assets"""
        asset_metrics = {}
        for timestamp, row in self.data.iterrows():
            for asset, value in row['asset_values'].items():
                if asset not in asset_metrics:
                    asset_metrics[asset] = []
                asset_metrics[asset].append(value)
        
        results = {}
        for asset, values in asset_metrics.items():
            values_series = pd.Series(values)
            returns = values_series.pct_change().dropna()
            
            results[asset] = {
                'total_return': (values[-1] / values[0] - 1) * 100,
                'volatility': returns.std() * np.sqrt(252) * 100,
                'weight': values[-1] / sum(self.data['asset_values'].iloc[-1].values()) * 100
            }
        
        return results

    def calculate_correlations(self) -> pd.DataFrame:
        """Calculate correlation matrix between assets"""
        asset_returns = {}
        for timestamp, row in self.data.iterrows():
            for asset, value in row['asset_values'].items():
                if asset not in asset_returns:
                    asset_returns[asset] = []
                asset_returns[asset].append(value)
        
        returns_df = pd.DataFrame(asset_returns).pct_change().dropna()
        return returns_df.corr()

    def generate_portfolio_report(self) -> Dict[str, Any]:
        """Generate comprehensive portfolio report"""
        returns = self.calculate_returns()
        risk_metrics = self.calculate_risk_metrics()
        asset_metrics = self.calculate_asset_metrics()
        correlations = self.calculate_correlations()
        
        # Calculate diversification score
        eigenvalues = np.linalg.eigvals(correlations)
        diversification_score = (1 - (max(eigenvalues) / sum(eigenvalues))) * 100
        
        # Calculate performance attribution
        total_return = returns['total_return']
        attribution = {}
        for asset, metrics in asset_metrics.items():
            attribution[asset] = {
                'contribution': (metrics['total_return'] * metrics['weight'] / 100),
                'weight': metrics['weight']
            }
        
        return {
            'summary': {
                'total_value': sum(self.data['asset_values'].iloc[-1].values()) + self.data['cash_value'].iloc[-1],
                'total_return': returns['total_return'],
                'volatility': returns['volatility'],
                'sharpe_ratio': risk_metrics['sharpe_ratio'],
                'max_drawdown': risk_metrics['max_drawdown']
            },
            'risk_metrics': risk_metrics,
            'asset_metrics': asset_metrics,
            'diversification': {
                'score': diversification_score,
                'correlations': correlations.to_dict()
            },
            'attribution': attribution,
            'recommendations': self._generate_recommendations(
                risk_metrics, asset_metrics, diversification_score
            )
        }

    def _generate_recommendations(self, risk_metrics: Dict[str, float],
                                asset_metrics: Dict[str, Dict[str, float]],
                                diversification_score: float) -> List[str]:
        """Generate portfolio recommendations based on analytics"""
        recommendations = []
        
        # Risk-based recommendations
        if risk_metrics['sharpe_ratio'] < 1:
            recommendations.append("Consider reducing portfolio risk or increasing expected returns")
        
        if risk_metrics['max_drawdown'] < -20:
            recommendations.append("Implement stronger stop-loss measures to limit drawdowns")
        
        # Diversification recommendations
        if diversification_score < 60:
            recommendations.append("Portfolio could benefit from increased diversification")
        
        # Asset-specific recommendations
        for asset, metrics in asset_metrics.items():
            if metrics['weight'] > 25:
                recommendations.append(f"Consider reducing exposure to {asset} (currently {metrics['weight']:.1f}%)")
            if metrics['volatility'] > 40:
                recommendations.append(f"High volatility detected in {asset}. Consider hedging or reducing position")
        
        return recommendations
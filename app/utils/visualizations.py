import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

class AdvancedVisualizations:
    @staticmethod
    def create_correlation_heatmap(returns_df: pd.DataFrame) -> Dict:
        """Create correlation heatmap for assets"""
        corr_matrix = returns_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        
        fig.update_layout(
            title='Asset Correlation Heatmap',
            xaxis_title='Assets',
            yaxis_title='Assets'
        )
        
        return fig.to_json()

    @staticmethod
    def create_returns_heatmap(returns_df: pd.DataFrame) -> Dict:
        """Create returns heatmap by time period"""
        # Resample returns to daily and pivot for heatmap
        daily_returns = returns_df.resample('D').sum()
        returns_by_weekday = daily_returns.groupby(
            [daily_returns.index.year, daily_returns.index.weekday]
        ).mean()
        
        pivot_table = returns_by_weekday.pivot_table(
            index=returns_by_weekday.index.get_level_values(0),
            columns=returns_by_weekday.index.get_level_values(1),
            values=returns_by_weekday.columns[0]
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
            y=pivot_table.index,
            colorscale='RdYlGn'
        ))
        
        fig.update_layout(
            title='Returns Heatmap by Weekday',
            xaxis_title='Weekday',
            yaxis_title='Year'
        )
        
        return fig.to_json()

    @staticmethod
    def create_drawdown_chart(equity_curve: pd.Series) -> Dict:
        """Create underwater (drawdown) chart"""
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve / rolling_max - 1
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdowns.index,
            y=drawdowns.values * 100,
            fill='tozeroy',
            name='Drawdown'
        ))
        
        fig.update_layout(
            title='Portfolio Drawdown',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            yaxis_tickformat='%'
        )
        
        return fig.to_json()

    @staticmethod
    def create_risk_contribution_chart(weights: np.array, covariance: np.array, 
                                     assets: List[str]) -> Dict:
        """Create risk contribution chart"""
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
        marginal_risk = np.dot(covariance, weights) / portfolio_risk
        risk_contribution = weights * marginal_risk
        
        fig = go.Figure(data=go.Pie(
            labels=assets,
            values=risk_contribution,
            hole=0.4
        ))
        
        fig.update_layout(
            title='Portfolio Risk Contribution'
        )
        
        return fig.to_json()

    @staticmethod
    def create_efficient_frontier(returns: pd.DataFrame, min_vol_portfolio: Dict,
                                max_sharpe_portfolio: Dict) -> Dict:
        """Create efficient frontier visualization"""
        # Generate portfolio combinations
        num_portfolios = 1000
        returns_array = []
        volatility_array = []
        
        for _ in range(num_portfolios):
            weights = np.random.random(len(returns.columns))
            weights = weights / np.sum(weights)
            
            portfolio_return = np.sum(returns.mean() * weights) * 252
            portfolio_std = np.sqrt(
                np.dot(weights.T, np.dot(returns.cov() * 252, weights))
            )
            
            returns_array.append(portfolio_return)
            volatility_array.append(portfolio_std)
        
        fig = go.Figure()
        
        # Plot random portfolios
        fig.add_trace(go.Scatter(
            x=volatility_array,
            y=returns_array,
            mode='markers',
            marker=dict(
                size=5,
                color='blue',
                opacity=0.5
            ),
            name='Possible Portfolios'
        ))
        
        # Plot minimum volatility and maximum Sharpe ratio portfolios
        fig.add_trace(go.Scatter(
            x=[min_vol_portfolio['volatility']],
            y=[min_vol_portfolio['return']],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='star'
            ),
            name='Minimum Volatility'
        ))
        
        fig.add_trace(go.Scatter(
            x=[max_sharpe_portfolio['volatility']],
            y=[max_sharpe_portfolio['return']],
            mode='markers',
            marker=dict(
                size=15,
                color='green',
                symbol='star'
            ),
            name='Maximum Sharpe Ratio'
        ))
        
        fig.update_layout(
            title='Efficient Frontier',
            xaxis_title='Volatility',
            yaxis_title='Expected Return',
            showlegend=True
        )
        
        return fig.to_json()

    @staticmethod
    def create_rolling_metrics(returns: pd.Series, window: int = 252) -> Dict:
        """Create rolling metrics visualization"""
        rolling_return = returns.rolling(window=window).mean() * window
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(window)
        rolling_sharpe = rolling_return / rolling_vol
        
        fig = make_subplots(rows=3, cols=1,
                           subplot_titles=('Rolling Return',
                                         'Rolling Volatility',
                                         'Rolling Sharpe Ratio'))
        
        fig.add_trace(
            go.Scatter(x=rolling_return.index, y=rolling_return.values,
                      name='Return'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol.values,
                      name='Volatility'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                      name='Sharpe Ratio'),
            row=3, col=1
        )
        
        fig.update_layout(height=900, title='Rolling Metrics')
        
        return fig.to_json()

    @staticmethod
    def create_factor_exposure_heatmap(factor_exposures: pd.DataFrame) -> Dict:
        """Create factor exposure heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=factor_exposures.values,
            x=factor_exposures.columns,
            y=factor_exposures.index,
            colorscale='RdYlBu'
        ))
        
        fig.update_layout(
            title='Factor Exposures',
            xaxis_title='Factors',
            yaxis_title='Assets'
        )
        
        return fig.to_json()

    @staticmethod
    def create_risk_metrics_radar(risk_metrics: Dict) -> Dict:
        """Create risk metrics radar chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=[risk_metrics[metric] for metric in risk_metrics.keys()],
            theta=list(risk_metrics.keys()),
            fill='toself',
            name='Risk Metrics'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False
        )
        
        return fig.to_json()

    @staticmethod
    def create_pnl_distribution(returns: pd.Series) -> Dict:
        """Create P&L distribution with fitted normal distribution"""
        fig = go.Figure()
        
        # Add histogram of returns
        fig.add_trace(go.Histogram(
            x=returns,
            name='P&L Distribution',
            histnorm='probability',
            nbinsx=50
        ))
        
        # Add normal distribution fit
        mu = returns.mean()
        sigma = returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            name='Normal Distribution Fit',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='P&L Distribution with Normal Fit',
            xaxis_title='Return',
            yaxis_title='Probability',
            bargap=0.1
        )
        
        return fig.to_json()

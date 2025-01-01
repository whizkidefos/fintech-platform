import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class Advanced3DVisualizations:
    @staticmethod
    def create_3d_efficient_frontier(returns: pd.DataFrame,
                                   portfolios: List[np.ndarray],
                                   metrics: Dict[str, List[float]]) -> Dict:
        """Create 3D efficient frontier visualization"""
        # Calculate portfolio metrics
        returns_array = np.array(metrics['returns'])
        risk_array = np.array(metrics['risks'])
        sharpe_array = np.array(metrics['sharpe_ratios'])
        
        # Create 3D scatter plot
        fig = go.Figure(data=[
            go.Scatter3d(
                x=returns_array,
                y=risk_array,
                z=sharpe_array,
                mode='markers',
                marker=dict(
                    size=8,
                    color=sharpe_array,
                    colorscale='Viridis',
                    opacity=0.8
                ),
                text=[f'Portfolio {i}' for i in range(len(portfolios))],
                hovertemplate=(
                    'Return: %{x:.2%}<br>' +
                    'Risk: %{y:.2%}<br>' +
                    'Sharpe Ratio: %{z:.2f}<br>' +
                    '%{text}'
                )
            )
        ])
        
        fig.update_layout(
            title='3D Efficient Frontier',
            scene=dict(
                xaxis_title='Expected Return',
                yaxis_title='Risk (Volatility)',
                zaxis_title='Sharpe Ratio'
            )
        )
        
        return fig.to_json()

    @staticmethod
    def create_3d_risk_decomposition(returns: pd.DataFrame,
                                   risk_contributions: np.ndarray,
                                   time_periods: List[str]) -> Dict:
        """Create 3D risk decomposition visualization"""
        assets = returns.columns
        periods = np.array(time_periods)
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(len(assets)),
                          np.arange(len(periods)))
        
        # Create 3D surface plot
        fig = go.Figure(data=[
            go.Surface(
                x=x,
                y=y,
                z=risk_contributions,
                colorscale='RdYlBu',
                hoverongaps=False,
                hovertemplate=(
                    'Asset: %{x}<br>' +
                    'Period: %{y}<br>' +
                    'Risk Contribution: %{z:.2%}'
                )
            )
        ])
        
        # Update layout
        fig.update_layout(
            title='3D Risk Decomposition Over Time',
            scene=dict(
                xaxis=dict(
                    title='Assets',
                    ticktext=list(assets),
                    tickvals=list(range(len(assets)))
                ),
                yaxis=dict(
                    title='Time Periods',
                    ticktext=list(periods),
                    tickvals=list(range(len(periods)))
                ),
                zaxis_title='Risk Contribution'
            )
        )
        
        return fig.to_json()

    @staticmethod
    def create_3d_correlation_network(returns: pd.DataFrame,
                                    threshold: float = 0.5) -> Dict:
        """Create 3D correlation network visualization"""
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Create network
        G = nx.Graph()
        
        # Add nodes
        for asset in returns.columns:
            G.add_node(asset)
        
        # Add edges for correlations above threshold
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    G.add_edge(corr_matrix.index[i], corr_matrix.columns[j],
                             weight=abs(corr_matrix.iloc[i, j]))
        
        # Calculate 3D layout
        pos = nx.spring_layout(G, dim=3)
        
        # Create edge trace
        edge_trace = []
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            weight = G[edge[0]][edge[1]]['weight']
            
            edge_trace.append(
                go.Scatter3d(
                    x=[x0, x1],
                    y=[y0, y1],
                    z=[z0, z1],
                    mode='lines',
                    line=dict(
                        width=2,
                        color=f'rgba(70, 130, 180, {weight})'
                    ),
                    hoverinfo='none'
                )
            )
        
        # Create node trace
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        node_size = []
        
        for node in G.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_text.append(node)
            node_size.append(G.degree(node) * 5)
        
        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            marker=dict(
                size=node_size,
                color=node_size,
                colorscale='Viridis',
                opacity=0.8
            ),
            hovertemplate='Asset: %{text}<br>Connections: %{marker.size}'
        )
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace])
        fig.update_layout(
            title='3D Correlation Network',
            showlegend=False,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        return fig.to_json()

    @staticmethod
    def create_3d_regime_visualization(returns: pd.DataFrame,
                                     regime_probabilities: np.ndarray,
                                     n_regimes: int) -> Dict:
        """Create 3D regime visualization using dimensionality reduction"""
        # Prepare features
        features = np.column_stack([
            returns.mean(axis=1),
            returns.std(axis=1),
            returns.skew(axis=1)
        ])
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=3)
        reduced_features = pca.fit_transform(features)
        
        # Create scatter plot for each regime
        traces = []
        for i in range(n_regimes):
            regime_mask = np.argmax(regime_probabilities, axis=1) == i
            
            traces.append(
                go.Scatter3d(
                    x=reduced_features[regime_mask, 0],
                    y=reduced_features[regime_mask, 1],
                    z=reduced_features[regime_mask, 2],
                    mode='markers',
                    name=f'Regime {i+1}',
                    marker=dict(
                        size=8,
                        opacity=0.7
                    ),
                    hovertemplate=(
                        'PC1: %{x:.2f}<br>' +
                        'PC2: %{y:.2f}<br>' +
                        'PC3: %{z:.2f}'
                    )
                )
            )
        
        # Create figure
        fig = go.Figure(data=traces)
        fig.update_layout(
            title='3D Market Regime Visualization',
            scene=dict(
                xaxis_title='First Principal Component',
                yaxis_title='Second Principal Component',
                zaxis_title='Third Principal Component'
            )
        )
        
        return fig.to_json()

class RealTimeMonitor:
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.price_buffer = pd.DataFrame()
        self.signal_buffer = pd.DataFrame()
        self.risk_buffer = pd.DataFrame()

    def update(self, new_data: Dict[str, pd.DataFrame]) -> Dict:
        """Update monitoring data and generate visualizations"""
        # Update buffers
        self.price_buffer = pd.concat(
            [self.price_buffer, new_data['prices']]
        ).tail(self.buffer_size)
        
        self.signal_buffer = pd.concat(
            [self.signal_buffer, new_data['signals']]
        ).tail(self.buffer_size)
        
        self.risk_buffer = pd.concat(
            [self.risk_buffer, new_data['risks']]
        ).tail(self.buffer_size)
        
        # Generate visualizations
        return {
            'price_chart': self._create_price_chart(),
            'signal_chart': self._create_signal_chart(),
            'risk_chart': self._create_risk_chart(),
            'alerts': self._generate_alerts()
        }

    def _create_price_chart(self) -> Dict:
        """Create real-time price chart"""
        fig = go.Figure()
        
        for asset in self.price_buffer.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.price_buffer.index,
                    y=self.price_buffer[asset],
                    name=asset,
                    mode='lines'
                )
            )
        
        fig.update_layout(
            title='Real-Time Asset Prices',
            xaxis_title='Time',
            yaxis_title='Price'
        )
        
        return fig.to_json()

    def _create_signal_chart(self) -> Dict:
        """Create real-time signal chart"""
        fig = go.Figure()
        
        for signal in self.signal_buffer.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.signal_buffer.index,
                    y=self.signal_buffer[signal],
                    name=signal,
                    mode='lines+markers'
                )
            )
        
        fig.update_layout(
            title='Real-Time Trading Signals',
            xaxis_title='Time',
            yaxis_title='Signal Strength'
        )
        
        return fig.to_json()

    def _create_risk_chart(self) -> Dict:
        """Create real-time risk chart"""
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('Portfolio Risk',
                                         'Risk Contributions'))
        
        # Portfolio risk
        fig.add_trace(
            go.Scatter(
                x=self.risk_buffer.index,
                y=self.risk_buffer['portfolio_risk'],
                name='Portfolio Risk',
                fill='tozeroy'
            ),
            row=1, col=1
        )
        
        # Risk contributions
        risk_cols = [col for col in self.risk_buffer.columns
                    if col != 'portfolio_risk']
        
        for col in risk_cols:
            fig.add_trace(
                go.Bar(
                    x=self.risk_buffer.index,
                    y=self.risk_buffer[col],
                    name=col
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title='Real-Time Risk Monitoring',
            height=800
        )
        
        return fig.to_json()

    def _generate_alerts(self) -> List[Dict]:
        """Generate alerts based on monitoring data"""
        alerts = []
        
        # Risk alerts
        latest_risk = self.risk_buffer['portfolio_risk'].iloc[-1]
        risk_ma = self.risk_buffer['portfolio_risk'].rolling(20).mean().iloc[-1]
        
        if latest_risk > risk_ma * 1.5:
            alerts.append({
                'type': 'risk',
                'level': 'high',
                'message': 'Portfolio risk significantly above average'
            })
        
        # Signal alerts
        for signal in self.signal_buffer.columns:
            latest_signal = self.signal_buffer[signal].iloc[-1]
            if abs(latest_signal) > 2:
                alerts.append({
                    'type': 'signal',
                    'level': 'medium',
                    'message': f'Strong {signal} signal detected'
                })
        
        # Price alerts
        for asset in self.price_buffer.columns:
            returns = self.price_buffer[asset].pct_change()
            latest_return = returns.iloc[-1]
            
            if abs(latest_return) > 0.02:
                alerts.append({
                    'type': 'price',
                    'level': 'medium',
                    'message': f'Large price move in {asset}'
                })
        
        return alerts

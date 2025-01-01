import numpy as np
import pandas as pd
import networkx as nx
import community
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

class NetworkVisualizations:
    @staticmethod
    def create_correlation_network(returns: pd.DataFrame,
                                 threshold: float = 0.5) -> Dict:
        """Create correlation network visualization"""
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
        
        # Calculate layout
        pos = nx.spring_layout(G)
        
        # Create visualization
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = G[edge[0]][edge[1]]['weight']
            
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    line=dict(width=weight * 3, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )
            )
        
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            text=list(G.nodes()),
            textposition='top center',
            marker=dict(
                size=20,
                color=list(nx.degree_centrality(G).values()),
                colorscale='Viridis',
                showscale=True
            )
        )
        
        fig = go.Figure(data=edge_trace + [node_trace])
        fig.update_layout(
            title='Asset Correlation Network',
            showlegend=False,
            hovermode='closest'
        )
        
        return fig.to_json()

    @staticmethod
    def create_minimum_spanning_tree(returns: pd.DataFrame) -> Dict:
        """Create minimum spanning tree visualization"""
        # Calculate distance matrix
        corr_matrix = returns.corr()
        dist_matrix = np.sqrt(2 * (1 - corr_matrix))
        
        # Create MST
        G = nx.from_numpy_array(dist_matrix.values)
        mst = nx.minimum_spanning_tree(G)
        
        # Calculate layout
        pos = nx.spring_layout(mst)
        
        # Create visualization
        edge_trace = []
        for edge in mst.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    line=dict(width=2, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )
            )
        
        node_trace = go.Scatter(
            x=[pos[node][0] for node in mst.nodes()],
            y=[pos[node][1] for node in mst.nodes()],
            mode='markers+text',
            text=[returns.columns[i] for i in mst.nodes()],
            textposition='top center',
            marker=dict(
                size=20,
                color=list(nx.degree_centrality(mst).values()),
                colorscale='Viridis',
                showscale=True
            )
        )
        
        fig = go.Figure(data=edge_trace + [node_trace])
        fig.update_layout(
            title='Minimum Spanning Tree of Assets',
            showlegend=False,
            hovermode='closest'
        )
        
        return fig.to_json()

class RegimeDetection:
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.scaler = StandardScaler()

    def detect_regimes_gmm(self, n_regimes: int = 3) -> Dict:
        """Detect market regimes using Gaussian Mixture Models"""
        # Prepare features
        features = self.scaler.fit_transform(
            np.column_stack([
                self.returns.mean(axis=1),
                self.returns.std(axis=1),
                self.returns.skew(axis=1)
            ])
        )
        
        # Fit GMM
        gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        regimes = gmm.fit_predict(features)
        probs = gmm.predict_proba(features)
        
        # Create visualization
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('Regime Probabilities',
                                         'Returns by Regime'))
        
        # Plot regime probabilities
        for i in range(n_regimes):
            fig.add_trace(
                go.Scatter(
                    x=self.returns.index,
                    y=probs[:, i],
                    name=f'Regime {i+1}',
                    stackgroup='one'
                ),
                row=1, col=1
            )
        
        # Plot returns colored by regime
        for i in range(n_regimes):
            mask = regimes == i
            fig.add_trace(
                go.Scatter(
                    x=self.returns.index[mask],
                    y=self.returns.mean(axis=1)[mask],
                    mode='markers',
                    name=f'Returns Regime {i+1}'
                ),
                row=2, col=1
            )
        
        fig.update_layout(height=800, title='Market Regimes Detection')
        
        return {
            'regimes': regimes.tolist(),
            'probabilities': probs.tolist(),
            'visualization': fig.to_json()
        }

    def detect_regimes_markov(self, n_regimes: int = 2) -> Dict:
        """Detect market regimes using Markov Switching Model"""
        # Fit Markov model
        model = MarkovRegression(
            self.returns.mean(axis=1),
            k_regimes=n_regimes,
            switching_variance=True
        ).fit()
        
        smoothed_probs = model.smoothed_marginal_probabilities
        regimes = np.argmax(smoothed_probs, axis=1)
        
        # Create visualization
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('Regime Probabilities',
                                         'Regime-Dependent Volatility'))
        
        # Plot regime probabilities
        for i in range(n_regimes):
            fig.add_trace(
                go.Scatter(
                    x=self.returns.index,
                    y=smoothed_probs[:, i],
                    name=f'Regime {i+1}',
                    stackgroup='one'
                ),
                row=1, col=1
            )
        
        # Plot regime-dependent volatility
        vol_by_regime = [
            self.returns.mean(axis=1)[regimes == i].std()
            for i in range(n_regimes)
        ]
        
        fig.add_trace(
            go.Bar(
                x=[f'Regime {i+1}' for i in range(n_regimes)],
                y=vol_by_regime,
                name='Volatility'
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=800, title='Markov Regime Switching Analysis')
        
        return {
            'regimes': regimes.tolist(),
            'probabilities': smoothed_probs.tolist(),
            'transition_matrix': model.transition_probabilities.tolist(),
            'visualization': fig.to_json()
        }

    def create_regime_summary(self, regimes: np.ndarray) -> Dict:
        """Create summary statistics for each regime"""
        summary = {}
        for i in np.unique(regimes):
            regime_data = self.returns.iloc[regimes == i]
            summary[f'regime_{i+1}'] = {
                'mean_return': float(regime_data.mean().mean()),
                'volatility': float(regime_data.std().mean()),
                'sharpe_ratio': float(regime_data.mean().mean() / regime_data.std().mean()),
                'duration': int(np.sum(regimes == i)),
                'transitions': int(np.sum(np.diff(regimes) == i))
            }
        return summary

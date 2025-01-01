import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
from datetime import datetime, timedelta

class DashboardGenerator:
    def __init__(self):
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ffbb33',
            'info': '#17a2b8',
            'background': '#f8f9fa',
            'text': '#212529'
        }
        
        self.chart_config = {
            'responsive': True,
            'displayModeBar': True,
            'displaylogo': False
        }

    def create_market_overview_dashboard(self, market_data: Dict) -> Dict:
        """Create market overview dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Asset Performance',
                'Correlation Matrix',
                'Volume Analysis',
                'Volatility Surface',
                'Market Depth',
                'Order Book'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'heatmap'}],
                [{'type': 'bar'}, {'type': 'surface'}],
                [{'type': 'scatter'}, {'type': 'scatter'}]
            ]
        )
        
        # Asset Performance
        for asset, data in market_data['prices'].items():
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data.values,
                    name=asset,
                    mode='lines'
                ),
                row=1, col=1
            )
        
        # Correlation Matrix
        fig.add_trace(
            go.Heatmap(
                z=market_data['correlation'],
                x=market_data['correlation'].index,
                y=market_data['correlation'].columns,
                colorscale='RdBu'
            ),
            row=1, col=2
        )
        
        # Volume Analysis
        fig.add_trace(
            go.Bar(
                x=market_data['volume'].index,
                y=market_data['volume'].values,
                name='Volume'
            ),
            row=2, col=1
        )
        
        # Volatility Surface
        fig.add_trace(
            go.Surface(
                z=market_data['volatility_surface'],
                colorscale='Viridis'
            ),
            row=2, col=2
        )
        
        # Market Depth
        fig.add_trace(
            go.Scatter(
                x=market_data['depth']['price'],
                y=market_data['depth']['cumulative_volume'],
                fill='tozeroy',
                name='Market Depth'
            ),
            row=3, col=1
        )
        
        # Order Book
        fig.add_trace(
            go.Scatter(
                x=market_data['orderbook']['price'],
                y=market_data['orderbook']['volume'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=market_data['orderbook']['side'],
                    colorscale=[[0, 'red'], [1, 'green']]
                ),
                name='Order Book'
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            height=1200,
            showlegend=True,
            paper_bgcolor=self.color_scheme['background'],
            plot_bgcolor=self.color_scheme['background'],
            font=dict(color=self.color_scheme['text'])
        )
        
        return fig.to_json()

    def create_portfolio_dashboard(self, portfolio_data: Dict) -> Dict:
        """Create portfolio analytics dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Portfolio Value',
                'Asset Allocation',
                'Risk Metrics',
                'Performance Attribution',
                'Drawdown Analysis',
                'Factor Exposure'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'pie'}],
                [{'type': 'bar'}, {'type': 'waterfall'}],
                [{'type': 'scatter'}, {'type': 'heatmap'}]
            ]
        )
        
        # Portfolio Value
        fig.add_trace(
            go.Scatter(
                x=portfolio_data['value'].index,
                y=portfolio_data['value'].values,
                fill='tozeroy',
                name='Portfolio Value'
            ),
            row=1, col=1
        )
        
        # Asset Allocation
        fig.add_trace(
            go.Pie(
                labels=portfolio_data['allocation'].index,
                values=portfolio_data['allocation'].values,
                hole=0.3
            ),
            row=1, col=2
        )
        
        # Risk Metrics
        fig.add_trace(
            go.Bar(
                x=portfolio_data['risk_metrics'].index,
                y=portfolio_data['risk_metrics'].values,
                name='Risk Metrics'
            ),
            row=2, col=1
        )
        
        # Performance Attribution
        fig.add_trace(
            go.Waterfall(
                x=portfolio_data['attribution'].index,
                y=portfolio_data['attribution'].values,
                connector={'line': {'color': 'rgb(63, 63, 63)'}}
            ),
            row=2, col=2
        )
        
        # Drawdown Analysis
        fig.add_trace(
            go.Scatter(
                x=portfolio_data['drawdown'].index,
                y=portfolio_data['drawdown'].values,
                fill='tozeroy',
                name='Drawdown'
            ),
            row=3, col=1
        )
        
        # Factor Exposure
        fig.add_trace(
            go.Heatmap(
                z=portfolio_data['factor_exposure'],
                x=portfolio_data['factor_exposure'].columns,
                y=portfolio_data['factor_exposure'].index,
                colorscale='RdYlBu'
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            height=1200,
            showlegend=True,
            paper_bgcolor=self.color_scheme['background'],
            plot_bgcolor=self.color_scheme['background'],
            font=dict(color=self.color_scheme['text'])
        )
        
        return fig.to_json()

    def create_ml_insights_dashboard(self, ml_data: Dict) -> Dict:
        """Create machine learning insights dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Model Predictions',
                'Attention Weights',
                'Generated Scenarios',
                'Feature Importance',
                'Model Performance',
                'Uncertainty Analysis'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'heatmap'}],
                [{'type': 'scatter'}, {'type': 'bar'}],
                [{'type': 'scatter'}, {'type': 'violin'}]
            ]
        )
        
        # Model Predictions
        fig.add_trace(
            go.Scatter(
                x=ml_data['predictions'].index,
                y=ml_data['predictions']['actual'],
                name='Actual',
                mode='lines'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=ml_data['predictions'].index,
                y=ml_data['predictions']['predicted'],
                name='Predicted',
                mode='lines'
            ),
            row=1, col=1
        )
        
        # Attention Weights
        fig.add_trace(
            go.Heatmap(
                z=ml_data['attention_weights'],
                colorscale='Viridis'
            ),
            row=1, col=2
        )
        
        # Generated Scenarios
        for scenario in ml_data['scenarios']:
            fig.add_trace(
                go.Scatter(
                    x=scenario.index,
                    y=scenario.values,
                    mode='lines',
                    opacity=0.3,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Feature Importance
        fig.add_trace(
            go.Bar(
                x=ml_data['feature_importance'].index,
                y=ml_data['feature_importance'].values,
                name='Feature Importance'
            ),
            row=2, col=2
        )
        
        # Model Performance
        fig.add_trace(
            go.Scatter(
                x=ml_data['performance'].index,
                y=ml_data['performance']['train_loss'],
                name='Training Loss'
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=ml_data['performance'].index,
                y=ml_data['performance']['val_loss'],
                name='Validation Loss'
            ),
            row=3, col=1
        )
        
        # Uncertainty Analysis
        fig.add_trace(
            go.Violin(
                y=ml_data['uncertainty'],
                box_visible=True,
                meanline_visible=True
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            height=1200,
            showlegend=True,
            paper_bgcolor=self.color_scheme['background'],
            plot_bgcolor=self.color_scheme['background'],
            font=dict(color=self.color_scheme['text'])
        )
        
        return fig.to_json()

    def create_real_time_dashboard(self) -> dash.Dash:
        """Create real-time monitoring dashboard"""
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.H1('Real-Time Market Monitor',
                   style={'textAlign': 'center'}),
            
            # Market Overview Section
            html.Div([
                html.H2('Market Overview'),
                dcc.Graph(id='price-chart'),
                dcc.Interval(
                    id='price-update',
                    interval=1000,
                    n_intervals=0
                )
            ]),
            
            # Trading Signals Section
            html.Div([
                html.H2('Trading Signals'),
                dcc.Graph(id='signal-chart'),
                dcc.Interval(
                    id='signal-update',
                    interval=5000,
                    n_intervals=0
                )
            ]),
            
            # Risk Monitoring Section
            html.Div([
                html.H2('Risk Monitor'),
                dcc.Graph(id='risk-chart'),
                dcc.Interval(
                    id='risk-update',
                    interval=10000,
                    n_intervals=0
                )
            ]),
            
            # Alerts Section
            html.Div([
                html.H2('Alerts'),
                html.Div(id='alerts-container'),
                dcc.Interval(
                    id='alerts-update',
                    interval=1000,
                    n_intervals=0
                )
            ])
        ])
        
        return app

    def create_custom_dashboard(self, components: List[Dict]) -> Dict:
        """Create custom dashboard with specified components"""
        num_rows = (len(components) + 1) // 2
        
        fig = make_subplots(
            rows=num_rows,
            cols=2,
            subplot_titles=[comp['title'] for comp in components],
            specs=[[{'type': comp['type']} for comp in components[i:i+2]]
                  for i in range(0, len(components), 2)]
        )
        
        for i, component in enumerate(components):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            if component['type'] == 'scatter':
                fig.add_trace(
                    go.Scatter(
                        x=component['data']['x'],
                        y=component['data']['y'],
                        name=component['name']
                    ),
                    row=row, col=col
                )
            elif component['type'] == 'bar':
                fig.add_trace(
                    go.Bar(
                        x=component['data']['x'],
                        y=component['data']['y'],
                        name=component['name']
                    ),
                    row=row, col=col
                )
            elif component['type'] == 'heatmap':
                fig.add_trace(
                    go.Heatmap(
                        z=component['data']['z'],
                        colorscale=component.get('colorscale', 'Viridis')
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=300 * num_rows,
            showlegend=True,
            paper_bgcolor=self.color_scheme['background'],
            plot_bgcolor=self.color_scheme['background'],
            font=dict(color=self.color_scheme['text'])
        )
        
        return fig.to_json()

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb

class HybridEnsemble(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int, num_models: int = 5):
        super().__init__()
        self.save_hyperparameters()
        
        # Deep Learning Models
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        
        # Attention Layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Output layers for each model
        self.lstm_out = nn.Linear(hidden_dim, 1)
        self.transformer_out = nn.Linear(hidden_dim, 1)
        self.gru_out = nn.Linear(hidden_dim, 1)
        
        # Ensemble weights
        self.ensemble_weights = nn.Parameter(
            torch.ones(num_models) / num_models
        )
        
        # Traditional ML Models
        self.traditional_models = {
            'rf': RandomForestRegressor(n_estimators=100),
            'gb': GradientBoostingRegressor(n_estimators=100),
            'cb': CatBoostRegressor(iterations=100, verbose=False),
            'lgb': lgb.LGBMRegressor(n_estimators=100),
            'xgb': xgb.XGBRegressor(n_estimators=100)
        }
        
        self.scaler = StandardScaler()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_out(lstm_out[:, -1, :])
        
        # Transformer
        transformer_out = self.transformer(x)
        transformer_out = self.transformer_out(transformer_out[:, -1, :])
        
        # GRU
        gru_out, _ = self.gru(x)
        gru_out = self.gru_out(gru_out[:, -1, :])
        
        # Combine predictions with learned weights
        ensemble_weights = F.softmax(self.ensemble_weights, dim=0)
        combined_pred = (
            ensemble_weights[0] * lstm_out +
            ensemble_weights[1] * transformer_out +
            ensemble_weights[2] * gru_out
        )
        
        return combined_pred

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                     batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def fit_traditional_models(self, X: np.ndarray, y: np.ndarray):
        """Fit traditional ML models"""
        X_scaled = self.scaler.fit_transform(X)
        
        for name, model in self.traditional_models.items():
            model.fit(X_scaled, y)

    def predict_traditional(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from traditional ML models"""
        X_scaled = self.scaler.transform(X)
        predictions = {}
        
        for name, model in self.traditional_models.items():
            predictions[name] = model.predict(X_scaled)
        
        return predictions

class AdaptiveHybridModel:
    def __init__(self, config: Dict):
        self.hybrid_ensemble = HybridEnsemble(**config['ensemble'])
        self.regime_detector = MarketRegimeDetector(**config['regime'])
        self.model_selector = ModelSelector(**config['selector'])
        
        self.current_regime = None
        self.active_models = []

    def update_regime(self, market_data: pd.DataFrame):
        """Update current market regime"""
        self.current_regime = self.regime_detector.detect_regime(market_data)
        self.active_models = self.model_selector.select_models(
            self.current_regime
        )

    def predict(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Make predictions using regime-appropriate models"""
        # Update regime if needed
        self.update_regime(market_data)
        
        predictions = {}
        confidences = {}
        
        # Get predictions from selected models
        for model_name in self.active_models:
            if model_name in self.hybrid_ensemble.traditional_models:
                pred = self.hybrid_ensemble.predict_traditional(
                    market_data.values
                )[model_name]
                predictions[model_name] = pred
                
                # Calculate confidence based on model performance in current regime
                confidences[model_name] = self.model_selector.get_confidence(
                    model_name,
                    self.current_regime
                )
            else:
                # Deep learning ensemble prediction
                with torch.no_grad():
                    pred = self.hybrid_ensemble(
                        torch.FloatTensor(market_data.values)
                    )
                predictions[model_name] = pred.numpy()
                confidences[model_name] = self.model_selector.get_confidence(
                    model_name,
                    self.current_regime
                )
        
        # Weighted ensemble prediction
        weights = np.array([confidences[model] for model in self.active_models])
        weights = weights / weights.sum()
        
        ensemble_prediction = np.zeros_like(
            list(predictions.values())[0]
        )
        
        for weight, (_, pred) in zip(weights, predictions.items()):
            ensemble_prediction += weight * pred
        
        return ensemble_prediction, {
            'predictions': predictions,
            'confidences': confidences,
            'regime': self.current_regime,
            'weights': weights
        }

class MarketRegimeDetector:
    def __init__(self, n_regimes: int = 3, window_size: int = 50):
        self.n_regimes = n_regimes
        self.window_size = window_size
        
        # Initialize models
        self.hmm = None
        self.gmm = None
        self.clustering = None

    def detect_regime(self, market_data: pd.DataFrame) -> str:
        """Detect current market regime using multiple methods"""
        # Calculate features
        features = self._calculate_features(market_data)
        
        # Get regime predictions from different models
        regime_predictions = {
            'hmm': self._hmm_regime(features),
            'gmm': self._gmm_regime(features),
            'clustering': self._clustering_regime(features)
        }
        
        # Ensemble regime prediction
        final_regime = self._ensemble_regime(regime_predictions)
        
        return final_regime

    def _calculate_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Calculate regime detection features"""
        returns = market_data.pct_change()
        
        features = np.column_stack([
            returns.mean(axis=1),
            returns.std(axis=1),
            returns.skew(axis=1),
            returns.kurt(axis=1)
        ])
        
        return features

    def _hmm_regime(self, features: np.ndarray) -> int:
        """Get regime prediction from HMM"""
        if self.hmm is None:
            from hmmlearn import hmm
            self.hmm = hmm.GaussianHMM(n_components=self.n_regimes)
            self.hmm.fit(features)
        
        return self.hmm.predict(features)[-1]

    def _gmm_regime(self, features: np.ndarray) -> int:
        """Get regime prediction from GMM"""
        if self.gmm is None:
            from sklearn.mixture import GaussianMixture
            self.gmm = GaussianMixture(n_components=self.n_regimes)
            self.gmm.fit(features)
        
        return self.gmm.predict(features.reshape(1, -1))[0]

    def _clustering_regime(self, features: np.ndarray) -> int:
        """Get regime prediction from clustering"""
        if self.clustering is None:
            from sklearn.cluster import KMeans
            self.clustering = KMeans(n_clusters=self.n_regimes)
            self.clustering.fit(features)
        
        return self.clustering.predict(features.reshape(1, -1))[0]

    def _ensemble_regime(self, predictions: Dict[str, int]) -> str:
        """Combine regime predictions"""
        # Simple majority voting
        regime = max(set(predictions.values()), key=list(predictions.values()).count)
        
        # Map regime to descriptive state
        regime_map = {
            0: 'low_volatility',
            1: 'high_volatility',
            2: 'transition'
        }
        
        return regime_map.get(regime, 'unknown')

class ModelSelector:
    def __init__(self, performance_window: int = 100):
        self.performance_window = performance_window
        self.performance_history = {}
        self.regime_performance = {}

    def select_models(self, regime: str) -> List[str]:
        """Select best performing models for current regime"""
        if regime not in self.regime_performance:
            # Default model selection if no performance history
            return ['ensemble', 'rf', 'lgb']
        
        # Get top 3 performing models for current regime
        performance = self.regime_performance[regime]
        top_models = sorted(
            performance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return [model for model, _ in top_models]

    def update_performance(self, model: str, regime: str,
                         performance: float):
        """Update model performance history"""
        if regime not in self.regime_performance:
            self.regime_performance[regime] = {}
        
        if model not in self.performance_history:
            self.performance_history[model] = []
        
        self.performance_history[model].append(performance)
        
        # Update regime-specific performance
        self.regime_performance[regime][model] = np.mean(
            self.performance_history[model][-self.performance_window:]
        )

    def get_confidence(self, model: str, regime: str) -> float:
        """Get model confidence score for current regime"""
        if regime not in self.regime_performance or \
           model not in self.regime_performance[regime]:
            return 1.0
        
        return self.regime_performance[regime][model]

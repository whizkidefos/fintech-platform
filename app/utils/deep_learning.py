import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, sequence_length: int):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length]
        return x, y

class AttentionLayer(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return torch.sum(x * attention_weights, dim=1)

class DeepMarketPredictor(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 dropout: float = 0.2):
        super().__init__()
        self.save_hyperparameters()
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention layer
        self.attention = AttentionLayer(hidden_dim, hidden_dim // 2)
        
        # Output layers
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Loss function
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attended = self.attention(lstm_out)
        
        # Prediction
        return self.predictor(attended)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

class DeepReinforcementTrader(pl.LightningModule):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.save_hyperparameters()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.gamma = 0.99
        self.eps = 1e-8

    def forward(self, state):
        return self.actor(state)

    def get_value(self, state):
        return self.critic(state)

    def training_step(self, batch, batch_idx):
        states, actions, rewards, next_states, dones = batch
        
        # Actor loss
        action_probs = self.actor(states)
        values = self.critic(states)
        next_values = self.critic(next_states)
        
        # Compute advantages
        advantages = rewards + self.gamma * next_values * (1 - dones) - values
        
        # Policy gradient loss
        selected_action_probs = torch.sum(action_probs * actions, dim=1)
        actor_loss = -torch.mean(torch.log(selected_action_probs + self.eps) * 
                               advantages.detach())
        
        # Critic loss
        critic_loss = torch.mean(advantages ** 2)
        
        # Combined loss
        loss = actor_loss + 0.5 * critic_loss
        
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0003)
        return optimizer

class MarketEnvironment:
    def __init__(self, data: pd.DataFrame, window_size: int = 50):
        self.data = data
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.portfolio_value = 1.0
        self.positions = np.zeros(len(self.data.columns))
        return self._get_state()

    def step(self, action):
        # Execute action
        old_portfolio_value = self.portfolio_value
        self.positions = action
        
        # Calculate returns
        returns = self.data.iloc[self.current_step].values
        portfolio_return = np.sum(self.positions * returns)
        self.portfolio_value *= (1 + portfolio_return)
        
        # Calculate reward
        reward = np.log(self.portfolio_value / old_portfolio_value)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data)
        
        return self._get_state(), reward, done

    def _get_state(self):
        # Get market state features
        market_data = self.data.iloc[self.current_step - self.window_size:
                                   self.current_step].values
        positions = self.positions.reshape(1, -1)
        portfolio_value = np.array([[self.portfolio_value]])
        
        return np.concatenate([market_data, positions, portfolio_value], axis=1)

class DeepLearningTrader:
    def __init__(self, predictor_config: Dict, trader_config: Dict):
        self.predictor = DeepMarketPredictor(**predictor_config)
        self.trader = DeepReinforcementTrader(**trader_config)
        self.scaler = StandardScaler()

    def train_predictor(self, data: pd.DataFrame, sequence_length: int,
                       batch_size: int = 32, num_epochs: int = 100):
        # Prepare data
        scaled_data = self.scaler.fit_transform(data)
        dataset = TimeSeriesDataset(scaled_data, sequence_length)
        train_size = int(0.8 * len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, len(dataset) - train_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=20),
                ModelCheckpoint(monitor='val_loss')
            ]
        )
        
        trainer.fit(self.predictor, train_loader, val_loader)

    def train_trader(self, data: pd.DataFrame, episodes: int = 1000):
        env = MarketEnvironment(data)
        
        for episode in range(episodes):
            state = env.reset()
            done = False
            
            while not done:
                # Get action from trader
                action_probs = self.trader(torch.FloatTensor(state))
                action = torch.multinomial(action_probs, 1).item()
                
                # Take step in environment
                next_state, reward, done = env.step(action)
                
                # Store transition and train
                self.trader.training_step((
                    torch.FloatTensor(state),
                    torch.FloatTensor([action]),
                    torch.FloatTensor([reward]),
                    torch.FloatTensor(next_state),
                    torch.FloatTensor([done])
                ), None)
                
                state = next_state

    def predict(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make market predictions and generate trading signals
        
        Returns:
            Tuple of (predictions, trading_signals)
        """
        # Scale data
        scaled_data = self.scaler.transform(market_data)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.predictor(
                torch.FloatTensor(scaled_data).unsqueeze(0)
            )
        
        # Generate trading signals
        state = self._prepare_state(market_data, predictions)
        action_probs = self.trader(torch.FloatTensor(state))
        
        return (
            self.scaler.inverse_transform(predictions.numpy()),
            action_probs.numpy()
        )

    def _prepare_state(self, market_data: pd.DataFrame,
                      predictions: torch.Tensor) -> np.ndarray:
        """Prepare state for trader"""
        market_features = market_data.values
        pred_features = predictions.numpy()
        return np.concatenate([market_features, pred_features], axis=1)

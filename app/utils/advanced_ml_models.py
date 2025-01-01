import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]

class MarketTransformer(pl.LightningModule):
    def __init__(self, input_dim: int, d_model: int, nhead: int,
                 num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.save_hyperparameters()
        
        # Model architecture
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_mask)
        return self.decoder(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                     batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

class TimeGAN(pl.LightningModule):
    def __init__(self, seq_len: int, feature_dim: int, hidden_dim: int,
                 latent_dim: int):
        super().__init__()
        self.save_hyperparameters()
        
        # Generator
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * feature_dim),
            nn.Tanh()
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(seq_len * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Supervisor (for temporal consistency)
        self.supervisor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * feature_dim)
        )
        
        # Embedding
        self.embedder = nn.Sequential(
            nn.Linear(seq_len * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        generated = self.generator(z)
        return generated.view(-1, self.seq_len, self.feature_dim)

    def adversarial_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(y_hat, y)

    def reconstruction_loss(self, x_hat: torch.Tensor,
                          x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x_hat, x)

    def training_step(self, batch: torch.Tensor,
                     batch_idx: int,
                     optimizer_idx: int) -> torch.Tensor:
        # Sample noise
        z = torch.randn(batch.size(0), self.latent_dim)
        z = z.type_as(batch)
        
        # Train Generator
        if optimizer_idx == 0:
            # Generate fake data
            generated = self(z)
            generated_flat = generated.view(generated.size(0), -1)
            
            # Discriminator output
            d_output = self.discriminator(generated_flat)
            
            # Adversarial loss
            g_loss = self.adversarial_loss(
                d_output,
                torch.ones(batch.size(0), 1).type_as(batch)
            )
            
            # Supervised loss
            embedded = self.embedder(generated_flat)
            supervised = self.supervisor(embedded)
            s_loss = self.reconstruction_loss(supervised, generated_flat)
            
            # Total generator loss
            total_loss = g_loss + 10 * s_loss
            
            self.log('g_loss', g_loss)
            self.log('s_loss', s_loss)
            return total_loss
        
        # Train Discriminator
        if optimizer_idx == 1:
            # Real data
            real_flat = batch.view(batch.size(0), -1)
            d_real = self.discriminator(real_flat)
            
            # Fake data
            generated = self(z)
            generated_flat = generated.view(generated.size(0), -1)
            d_fake = self.discriminator(generated_flat.detach())
            
            # Adversarial loss
            real_loss = self.adversarial_loss(
                d_real,
                torch.ones(batch.size(0), 1).type_as(batch)
            )
            fake_loss = self.adversarial_loss(
                d_fake,
                torch.zeros(batch.size(0), 1).type_as(batch)
            )
            
            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            
            self.log('d_loss', d_loss)
            return d_loss

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(
            list(self.generator.parameters()) +
            list(self.supervisor.parameters()) +
            list(self.embedder.parameters()),
            lr=0.0002
        )
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
        return [g_opt, d_opt], []

class AdvancedMarketPredictor:
    def __init__(self, config: Dict):
        self.transformer = MarketTransformer(**config['transformer'])
        self.gan = TimeGAN(**config['gan'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.transformer.to(self.device)
        self.gan.to(self.device)

    def train_transformer(self, train_loader: DataLoader,
                         val_loader: DataLoader,
                         max_epochs: int = 100) -> None:
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
            accelerator='auto'
        )
        trainer.fit(self.transformer, train_loader, val_loader)

    def train_gan(self, train_loader: DataLoader,
                  max_epochs: int = 100) -> None:
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator='auto'
        )
        trainer.fit(self.gan, train_loader)

    def predict(self, market_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions using both transformer and GAN
        
        Returns:
            Tuple of (transformer_predictions, generated_scenarios)
        """
        self.transformer.eval()
        self.gan.eval()
        
        with torch.no_grad():
            # Transformer predictions
            transformer_pred = self.transformer(market_data.to(self.device))
            
            # Generate scenarios using GAN
            z = torch.randn(market_data.size(0), self.gan.latent_dim,
                          device=self.device)
            generated_scenarios = self.gan(z)
        
        return transformer_pred.cpu(), generated_scenarios.cpu()

    def generate_scenarios(self, num_scenarios: int,
                         condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate market scenarios using the GAN"""
        self.gan.eval()
        
        with torch.no_grad():
            z = torch.randn(num_scenarios, self.gan.latent_dim,
                          device=self.device)
            if condition is not None:
                z = torch.cat([z, condition.to(self.device)], dim=1)
            scenarios = self.gan(z)
        
        return scenarios.cpu()

    def get_attention_weights(self, market_data: torch.Tensor) -> torch.Tensor:
        """Get attention weights from transformer for interpretability"""
        self.transformer.eval()
        
        with torch.no_grad():
            # Forward pass through embedding and positional encoding
            x = self.transformer.embedding(market_data.to(self.device))
            x = self.transformer.pos_encoder(x)
            
            # Get attention weights from first layer
            attention_weights = self.transformer.transformer_encoder.layers[0].self_attn(
                x, x, x
            )[1]
        
        return attention_weights.cpu()

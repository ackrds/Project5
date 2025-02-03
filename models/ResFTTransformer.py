import torch
import torch.nn as nn

from mambular.arch_utils.get_norm_fn import get_normalization_layer
from mambular.arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from mambular.arch_utils.mlp_utils import MLPhead
from mambular.base_models import BaseModel

class ResidualAttentionLayer(nn.Module):
    """Custom Transformer Encoder Layer with Residual Attention mechanisms."""
    
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.nhead = config.n_heads
        
        # Multi-head attention with residual connection
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.nhead,
            dropout=config.attn_dropout,
            batch_first=True    
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, config.transformer_dim_feedforward),
            nn.ReLU(),
            nn.Dropout(config.ff_dropout),
            nn.Linear(config.transformer_dim_feedforward, self.d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(config.attn_dropout)
        self.dropout2 = nn.Dropout(config.attn_dropout)
        
        # Residual attention weights
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
    
    def forward(self, x, attention_mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, attention_weights = self.self_attn(
            x, x, x,
            attn_mask=attention_mask,
            need_weights=True
        )
        
        # Residual connection with learnable weight
        x = x + self.alpha * self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed-forward network with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.beta * self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x, attention_weights

class ResidualAttentionTransformer(BaseModel):
    """A Feature Transformer model with Residual Attention mechanisms for tabular data."""
    
    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes=1,
        config=None,
        **kwargs
    ):
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])
        self.returns_ensemble = False
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        
        # Embedding layer
        self.embedding_layer = EmbeddingLayer(
            num_feature_info=num_feature_info,
            cat_feature_info=cat_feature_info,
            config=config
        )
        
        # Create stack of residual attention layers
        self.attention_layers = nn.ModuleList([
            ResidualAttentionLayer(config)
            for _ in range(self.hparams.n_layers)
        ])
        
        # Final normalization
        self.norm_f = get_normalization_layer(config)
        
        # Output head
        self.tabular_head = MLPhead(
            input_dim=self.hparams.d_model,
            config=config,
            output_dim=num_classes
        )
        
        # Initialize pooling
        n_inputs = len(num_feature_info) + len(cat_feature_info)
        self.initialize_pooling_layers(config=config, n_inputs=n_inputs)
        
        # Store attention weights for visualization/analysis
        self.attention_weights = []
    
    def forward(self, num_features, cat_features):
        """Forward pass with residual attention mechanisms."""
        # Reset attention weights
        self.attention_weights = []
        
        # Initial embedding
        x = self.embedding_layer(num_features, cat_features)
        
        # Process through residual attention layers
        for layer in self.attention_layers:
            x, attn_weights = layer(x)
            self.attention_weights.append(attn_weights)
        
        # Pool sequence
        x = self.pool_sequence(x)
        
        # Final normalization if specified
        if self.norm_f is not None:
            x = self.norm_f(x)
            
        # Generate predictions
        preds = self.tabular_head(x)
        
        return preds
    
    def get_attention_maps(self):
        """Return attention weights for visualization."""
        return self.attention_weights

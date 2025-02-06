import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from mambular.arch_utils.get_norm_fn import get_normalization_layer
from mambular.arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from mambular.arch_utils.mlp_utils import MLPhead
from mambular.base_models import BaseModel
from mambular.configs import DefaultFTTransformerConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from mambular.arch_utils.get_norm_fn import get_normalization_layer
from mambular.arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from mambular.arch_utils.mlp_utils import MLPhead
from mambular.base_models import BaseModel
from mambular.configs import DefaultFTTransformerConfig

class ReGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)

class FastformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = getattr(config, "d_model", 128)
        self.num_heads = getattr(config, "n_heads", 8)
        self.head_dim = self.d_model // self.num_heads
        self.scale = math.sqrt(self.head_dim)
        self.bias = getattr(config, "bias", True)
        self.dropout_prob = getattr(config, "attn_dropout", 0.1)
        
        # Linear projections
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model)
        self.output_proj = nn.Linear(self.d_model, self.d_model)
        
        # Parameter vectors for additive attention
        self.query_weights = nn.Parameter(torch.zeros(self.num_heads, self.d_model))
        self.key_weights = nn.Parameter(torch.zeros(self.num_heads, self.d_model))
        
        # Normalization and dropout
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(self.dropout_prob)
        
        # Feed-forward network
        self.ff_dim = getattr(config, "transformer_dim_feedforward", 2048)
        self.linear1 = nn.Linear(self.d_model, self.ff_dim)
        self.linear2 = nn.Linear(self.ff_dim // 2, self.d_model)
        self.activation = ReGLU()
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.query_weights)
        nn.init.xavier_uniform_(self.key_weights)

    def forward(self, hidden_states, attention_mask=None):
        B, N, C = hidden_states.shape
        
        # Apply first normalization
        h = self.norm1(hidden_states)
        
        # Project to q, k, v
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, num_heads, N, head_dim]
        
        # Global query attention
        query_attn = torch.einsum('bhnd,hn->bhnd', q, self.query_weights) / self.scale
        if attention_mask is not None:
            query_attn = query_attn.masked_fill(attention_mask.unsqueeze(1).unsqueeze(-1), -1e9)
        query_weight = F.softmax(query_attn, dim=2)
        global_query = (query_weight * q).sum(dim=2, keepdim=True)  # [B, num_heads, 1, head_dim]
        
        # Key-Query interaction
        key_attn = torch.einsum('bhnd,hn->bhnd', k, self.key_weights) / self.scale
        if attention_mask is not None:
            key_attn = key_attn.masked_fill(attention_mask.unsqueeze(1).unsqueeze(-1), -1e9)
        key_weight = F.softmax(key_attn, dim=2)
        global_key = (key_weight * k).sum(dim=2, keepdim=True)  # [B, num_heads, 1, head_dim]
        
        # Combine with values
        out = v * global_key
        
        # Reshape and project output
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.output_proj(out)
        out = self.dropout(out)
        
        # First residual connection
        hidden_states = hidden_states + out
        
        # Feed-forward network
        h = self.norm2(hidden_states)
        h = self.linear2(self.activation(self.linear1(h)))
        h = self.dropout(h)
        
        # Second residual connection
        hidden_states = hidden_states + h
        
        return hidden_states

class Fastformer(BaseModel):
    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes=1,
        config=None,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])
        self.returns_ensemble = False
        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info

        # embedding layer
        self.embedding_layer = EmbeddingLayer(
            num_feature_info=num_feature_info,
            cat_feature_info=cat_feature_info,
            config=config,
        )

        # Create encoder layers
        self.encoder_layers = nn.ModuleList([
            FastformerEncoderLayer(config)
            for _ in range(self.hparams.n_layers)
        ])
        
        # Final normalization
        self.norm_f = get_normalization_layer(config)

        # Output head
        self.tabular_head = MLPhead(
            input_dim=self.hparams.d_model,
            config=config,
            output_dim=num_classes,
        )

        # pooling
        n_inputs = len(num_feature_info) + len(cat_feature_info)
        self.initialize_pooling_layers(config=config, n_inputs=n_inputs)
        
    def forward(self, num_features, cat_features):
        x = self.embedding_layer(num_features, cat_features)
        
        for layer in self.encoder_layers:
            x = layer(x)
        
        x = self.pool_sequence(x)
        
        if self.norm_f is not None:
            x = self.norm_f(x)
            
        preds = self.tabular_head(x)
        return preds

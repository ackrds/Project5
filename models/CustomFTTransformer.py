import torch
import torch.nn as nn
import torch.nn.functional as F

from mambular.arch_utils.get_norm_fn import get_normalization_layer
from mambular.arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from mambular.arch_utils.mlp_utils import MLPhead
from mambular.base_models import BaseModel
from mambular.configs.fttransformer_config import DefaultFTTransformerConfig



def reglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


class ReGLU(nn.Module):
    def forward(self, x):
        return reglu(x)


class GLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if x.size(-1) % 2 != 0:
            raise ValueError("Input dimension must be even")
        split_dim = x.size(-1) // 2
        return x[..., :split_dim] * torch.sigmoid(x[..., split_dim:])


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, config):
        super().__init__(
            d_model=getattr(config, "d_model", 128),
            nhead=getattr(config, "n_heads", 8),
            dim_feedforward=getattr(config, "transformer_dim_feedforward", 2048),
            dropout=getattr(config, "attn_dropout", 0.1),
            activation=getattr(config, "transformer_activation", F.relu),
            layer_norm_eps=getattr(config, "layer_norm_eps", 1e-5),
            norm_first=getattr(config, "norm_first", False),
        )
        self.bias = getattr(config, "bias", True)
        # self.custom_activation = getattr(config, "transformer_activation", F.relu)
        self.custom_activation = ReGLU()
        self.transformer_dim_feedforward = getattr(config, "transformer_dim_feedforward", 2048)
        self.d_model = getattr(config, "d_model", 128)

        # Additional setup based on the activation function
        if self.custom_activation in [ReGLU, GLU] or isinstance(self.custom_activation, ReGLU | GLU):
            self.linear1 = nn.Linear(
                self.d_model,
                self.transformer_dim_feedforward ,
                bias=self.bias,
            )
            self.linear2 = nn.Linear(
                self.transformer_dim_feedforward//2,
                self.d_model,
                bias=self.bias,
            )

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Use the provided activation function
        if self.custom_activation in [ReGLU, GLU] or isinstance(self.custom_activation, ReGLU | GLU):
            src2 = self.linear2(self.custom_activation(self.linear1(src)))
        else:
            src2 = self.linear2(self.custom_activation(self.linear1(src)))

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class DenseTransformerEncoderLayer(CustomTransformerEncoderLayer):
    def __init__(self, config, in_channels):
        super().__init__(config)
        self.in_channels = in_channels
        self.out_channels = getattr(config, "d_model", 128)
        
        # Downsample layer to handle concatenated features
        self.downsample = nn.Linear(
            self.in_channels,
            self.out_channels,
            bias=getattr(config, "bias", True)
        )
    
    def forward(self, src, prev_outputs=None, src_mask=None, src_key_padding_mask=None):
        # If we have previous outputs, concatenate them along the feature dimension
        if prev_outputs is not None:
            concatenated = torch.cat([src] + prev_outputs, dim=-1)
            # Downsample the concatenated features
            src = self.downsample(concatenated)
        
        # Standard transformer processing
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, 
                            key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        if self.custom_activation in [ReGLU, GLU] or isinstance(self.custom_activation, (ReGLU, GLU)):
            src2 = self.linear2(self.custom_activation(self.linear1(src)))
        else:
            src2 = self.linear2(self.custom_activation(self.linear1(src)))

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src
    
    
class FTTransformer(BaseModel):
    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes=1,
        config: DefaultFTTransformerConfig = DefaultFTTransformerConfig(),  # noqa: B008
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

        self.tabular_head = MLPhead(
            input_dim=self.hparams.d_model,
            config=config,
            output_dim=num_classes,
        )

        # pooling
        n_inputs = len(num_feature_info) + len(cat_feature_info)
        self.initialize_pooling_layers(config=config, n_inputs=n_inputs)

        
        # Replace the standard encoder with a dense version
        self.encoder_layers = nn.ModuleList()
        d_model = self.hparams.d_model
        
        # Create dense encoder layers
        for i in range(self.hparams.n_layers):
            # Calculate input channels for each layer (original + all previous outputs)
            in_channels = d_model * (i + 1)
            layer = DenseTransformerEncoderLayer(config=config, in_channels=in_channels)
            self.encoder_layers.append(layer)
            
        # Final normalization
        self.norm_f = get_normalization_layer(config)
        
    def forward(self, num_features, cat_features):
        x = self.embedding_layer(num_features, cat_features)
        
        # Store all intermediate outputs for dense connections
        layer_outputs = []
        current_features = x
        
        # Process through dense encoder layers
        for layer in self.encoder_layers:
            output = layer(current_features, 
                         prev_outputs=layer_outputs if layer_outputs else None)
            layer_outputs.append(output)
            current_features = output
        
        # Pool the final output
        x = self.pool_sequence(current_features)
        
        if self.norm_f is not None:
            x = self.norm_f(x)
            
        preds = self.tabular_head(x)
        return preds

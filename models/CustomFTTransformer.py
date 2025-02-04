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


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, n_features, config):
        """DenseTransformerEncoderLayer combines dense connectivity with transformer architecture.
        Each layer receives concatenated features from all previous layers.

        Args:
            n_features: Number of input features
            config: Configuration object with transformer parameters
        """
        super().__init__()
        self.d_model = getattr(config, "d_model", 128)
        n_layers = getattr(config, "n_layers", 6)
        n_heads = getattr(config, "n_heads", 8)
        attn_dropout = getattr(config, "attn_dropout", 0.2)
        ff_dropout = getattr(config, "ff_dropout", 0.2)
        activation = ReGLU()
        expansion_factor = 2 if isinstance(activation, (ReGLU, GLU)) else 4
        self.transformer_dim_feedforward = getattr(config, "transformer_dim_feedforward", 1024)
        self.layers = nn.ModuleList([])
        print(config.ff_dropout)
        
        for i in range(n_layers):
            # Calculate input dimension for current layer (d_model * (i+1))
            # because it receives concatenated features from all previous layers
            
            self.layers.append(nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=self.d_model,
                    num_heads=n_heads,
                    dropout=attn_dropout,
                    batch_first=True
                ),
                nn.Dropout(ff_dropout),
                nn.LayerNorm(self.d_model),
                
                nn.Sequential(
                    nn.Linear(self.d_model, self.transformer_dim_feedforward*2),
                    activation,
                    nn.Linear(self.transformer_dim_feedforward, self.d_model)
                ),
                nn.LayerNorm(self.d_model),
                nn.Dropout(ff_dropout),

            ]))

    def forward(self, x):
        """
        Args:
            x: Input embeddings (batch_size, seq_len, d_model)
        Returns:
            x: Transformed embeddings (batch_size, seq_len, d_model)
        """
        
        for  attn, norm1, dropout1,ffn, norm2, dropout2 in self.layers:
            x_attn = attn(x, x, x)[0]
            x = x + dropout1(x_attn)
            x = norm1(x)
            x_ffn = ffn(x)
            x = x + dropout2(x_ffn)
            x = norm2(x)
        return x
    
class CustomFTTransformer(BaseModel):
    """A Feature Transformer model for tabular data with categorical and numerical features, using embedding,
    transformer encoding, and pooling to produce final predictions.

    Parameters
    ----------
    cat_feature_info : dict
        Dictionary containing information about categorical features, including their names and dimensions.
    num_feature_info : dict
        Dictionary containing information about numerical features, including their names and dimensions.
    num_classes : int, optional
        The number of output classes or target dimensions for regression, by default 1.
    config : DefaultSAINTConfig, optional
        Configuration object containing model hyperparameters such as dropout rates, hidden layer sizes,
        transformer settings, and other architectural configurations, by default DefaultSAINTConfig().
    **kwargs : dict
        Additional keyword arguments for the BaseModel class.

    Attributes
    ----------
    pooling_method : str
        The pooling method to aggregate features after transformer encoding.
    cat_feature_info : dict
        Stores categorical feature information.
    num_feature_info : dict
        Stores numerical feature information.
    embedding_layer : EmbeddingLayer
        Layer for embedding categorical and numerical features.
    norm_f : nn.Module
        Normalization layer for the transformer output.
    encoder : nn.TransformerEncoder
        Transformer encoder for sequential processing of embedded features.
    tabular_head : MLPhead
        MLPhead layer to produce the final prediction based on the output of the transformer encoder.

    Methods
    -------
    forward(num_features, cat_features)
        Perform a forward pass through the model, including embedding, transformer encoding,
        pooling, and prediction steps.
    """

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
        n_inputs = len(num_feature_info) + len(cat_feature_info)
        if getattr(config, "use_cls", True):
            n_inputs += 1

        # embedding layer
        self.embedding_layer = EmbeddingLayer(
            num_feature_info=num_feature_info,
            cat_feature_info=cat_feature_info,
            config=config,
        )

        # transformer encoder
        self.norm_f = get_normalization_layer(config)
        self.encoder = CustomTransformerEncoderLayer(
            config=config,
            n_features=n_inputs,
        )

        self.tabular_head = MLPhead(
            input_dim=self.hparams.d_model,
            config=config,
            output_dim=num_classes,
        )

        # pooling

        self.initialize_pooling_layers(config=config, n_inputs=n_inputs)

    def forward(self, num_features, cat_features):
        """Defines the forward pass of the model.

        Parameters
        ----------
        num_features : Tensor
            Tensor containing the numerical features.
        cat_features : Tensor
            Tensor containing the categorical features.

        Returns
        -------
        Tensor
            The output predictions of the model.
        """
        x = self.embedding_layer(num_features, cat_features)

        x = self.encoder(x)

        x = self.pool_sequence(x)

        if self.norm_f is not None:
            x = self.norm_f(x)
        preds = self.tabular_head(x)

        return preds
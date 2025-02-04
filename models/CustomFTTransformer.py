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

    
class FTTransformer(BaseModel):
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
    config : DefaultFTTransformerConfig, optional
        Configuration object containing model hyperparameters such as dropout rates, hidden layer sizes,
        transformer settings, and other architectural configurations, by default DefaultFTTransformerConfig().
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

        # embedding layer
        self.embedding_layer = EmbeddingLayer(
            num_feature_info=num_feature_info,
            cat_feature_info=cat_feature_info,
            config=config,
        )

        # transformer encoder
        self.norm_f = get_normalization_layer(config)
        encoder_layer = CustomTransformerEncoderLayer(config=config)
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.hparams.n_layers,
            norm=self.norm_f,
        )

        self.tabular_head = MLPhead(
            input_dim=self.hparams.d_model,
            config=config,
            output_dim=num_classes,
        )

        # pooling
        n_inputs = len(num_feature_info) + len(cat_feature_info)
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
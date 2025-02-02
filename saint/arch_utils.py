import torch
import torch.nn as nn
from einops import rearrange

class RowColTransformer(nn.Module):
    def __init__(self, n_features, config):
        """RowColTransformer initialized with a configuration object.

        Args:
        - config: A configuration object containing all hyperparameters.
          Expected attributes:
            - d_model: Embedding dimension.
            - n_features: Number of features.
            - n_layers: Number of transformer layers.
            - n_heads: Number of attention heads.
            - dim_head: Dimension per head.
            - attn_dropout: Dropout rate for attention layers.
            - ff_dropout: Dropout rate for feedforward layers.
            - style: Transformer style ('col' or 'colrow').
        """

        super().__init__()
        d_model = getattr(config, "d_model", 128)
        n_layers = getattr(config, "n_layers", 6)
        n_heads = getattr(config, "n_heads", 8)
        attn_dropout = getattr(config, "attn_dropout", 0.1)
        ff_dropout = getattr(config, "ff_dropout", 0.1)
        activation = getattr(config, "activation", nn.GELU())

        self.n_features = n_features
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.LayerNorm(d_model),
                            nn.MultiheadAttention(
                                embed_dim=d_model,
                                num_heads=n_heads,
                                dropout=attn_dropout,
                                batch_first=True,
                            ),
                            nn.Dropout(ff_dropout),
                        ),
                        nn.Sequential(
                            nn.LayerNorm(d_model),
                            nn.Sequential(
                                nn.Linear(d_model, d_model * 4),
                                activation,
                                nn.Dropout(ff_dropout),
                                nn.Linear(d_model * 4, d_model),
                            ),
                        ),
                        nn.Sequential(
                            nn.LayerNorm(d_model * n_features),
                            nn.MultiheadAttention(
                                embed_dim=d_model * n_features,
                                num_heads=n_heads,
                                dropout=attn_dropout,
                                batch_first=True,
                            ),
                            nn.Dropout(ff_dropout),
                        ),
                        nn.Sequential(
                            nn.LayerNorm(d_model * n_features),
                            nn.Sequential(
                                nn.Linear(d_model * n_features, d_model * n_features * 4),
                                activation,
                                nn.Dropout(ff_dropout),
                                nn.Linear(d_model * n_features * 4, d_model * n_features),
                            ),
                        ),
                    ]
                )
            )



    def forward(self, x):
        """
        Args:
            x: Input embeddings of shape (N, J, D),
               where N = batch size, J = number of features, D = embedding dimension.
        """
        batch_size, n, d = x.shape
                        
        causal_mask = torch.triu(torch.ones(batch_size, batch_size), diagonal=1).bool()
        if x.is_cuda:
            causal_mask = causal_mask.cuda()

        for attn1, ff1, attn2, ff2 in self.layers:  # type: ignore
            x = rearrange(x, "b n d -> 1 b (n d)")
            x = attn2[1](
                x, x, x,
                attn_mask=causal_mask,
                need_weights=False
            )[0] + x  # Causal multihead attention with residual
            x = ff2(x) + x  # Feedforward with residual
            x = rearrange(x, "1 b (n d) -> b n d", n=n)

        return x
import torch.nn as nn
def output_head(output_dim, hidden_dim, dropout, n_classes):
    return nn.Sequential(
                    nn.Linear(output_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.SELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, n_classes)  # Predict both mean and variance
        ) 
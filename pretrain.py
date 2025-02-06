from typing import List, Dict
import torch
import numpy as np
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from mambular.base_models import BaseModel
from mambular.configs import DefaultMambularConfig, DefaultFTTransformerConfig
import torch.nn.functional as  F
from models.output_head import output_head

class PretrainingModel(BaseModel):
    def __init__(
            self,
            cat_feature_info,
            num_feature_info,
            model,
            output_dim,
            pretrain_config,
            config=DefaultFTTransformerConfig(),
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.temperature = getattr(pretrain_config, "temperature", 0.07)
        self.dropout = getattr(pretrain_config, "dropout", 0.2)
        self.hidden_dim = getattr(pretrain_config, "hidden_dim", 1024)
        self.projection_dim = getattr(pretrain_config, "projection_dim", 512)
        self.lambda_ = getattr(pretrain_config, "lambda_", 0.75)
        self.numeric_loss_type = getattr(pretrain_config, "numeric_loss_type", "nll")

        self.model = model(cat_feature_info, num_feature_info, output_dim, config=config)

        self.norm_f = self.model.norm_f
        self.embedding_layer = self.model.embedding_layer
        self.encoder = self.model.encoder
        self.tabular_head = self.model.tabular_head

        self.output_dim = output_dim

        # Improved projection head with larger capacity and bottleneck
        self.projector = nn.Sequential(
            nn.Linear(output_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),  # One more layer for deeper representation
            nn.LayerNorm(self.hidden_dim),
            nn.SELU(),
            nn.Linear(self.hidden_dim, self.projection_dim)  # Project to smaller dim for contrastive learning
        )
        # Define prediction heads
        self.num_heads = nn.ModuleList([
            output_head(output_dim, self.hidden_dim, self.dropout, 2) for _ in num_feature_info
        ])
        
        # Improved categorical prediction heads
        # Categorical prediction heads for InfoNCE
        self.cat_heads = nn.ModuleList([
            output_head(output_dim, self.hidden_dim, self.dropout, info["categories"]) 
            for info in cat_feature_info.values()
        ])    
    

    def manifold_mixup(self, embeddings):
        shuffled_embeddings = embeddings[torch.randperm(embeddings.size(0))]
        mixed_embeddings = self.lambda_ * embeddings + (1 - self.lambda_) * shuffled_embeddings
        
        return mixed_embeddings

    def forward(self, num_features, cat_features):
        # Get base encoding
        embedding = self.embedding_layer(num_features, cat_features)
        embedding = self.manifold_mixup(embedding)
        encoded = self.encoder(embedding)

        # Apply stochastic depth during training
        # if self.training:
        #     encoded = F.dropout2d(encoded.unsqueeze(-1), p=0.1, training=True).squeeze(-1)

        # lets keep this manually for now i will figure out later
        encoded = encoded.mean(dim=1)

        if self.norm_f is not None:
            encoded = self.norm_f(encoded)

        encoded = self.tabular_head(encoded)

        # Project for contrastive learning
        proj = self.projector(encoded)
        
        # Get predictions
        num_preds = [head(encoded) for head in self.num_heads]
        cat_logits = [head(encoded) for head in self.cat_heads]
        
        return {
            'encoded': encoded,
            'proj': proj,
            'num_preds': num_preds,
            'cat_logits': cat_logits
        }
    def compute_numeric_loss(self, num_preds, num_targets):
        """Compute reconstruction loss for numeric features"""
        num_recon_loss = 0.0
        for pred, target in zip(num_preds, num_targets):
            if self.numeric_loss_type == "mae":
                num_recon_loss += torch.mean(torch.abs(pred - target))
            elif self.numeric_loss_type == "nll":
                mean, log_var = pred.chunk(2, dim=-1)
                var = torch.exp(log_var)
                num_recon_loss += torch.mean(0.5 * (
                    torch.log(var) + 
                    (target.view(-1, 1) - mean)**2 / var
                ))
        return num_recon_loss

    def compute_categorical_loss(self, cat_logits, cat_targets):
        """Compute InfoNCE loss for categorical features"""
        cat_loss = 0.0
        for logits, target in zip(cat_logits, cat_targets):
            # Get similarities
            similarities = F.softmax(logits, dim=1)
            target_idx = target.view(-1, 1).long()
            
            # Safety check
            if target_idx.max() >= logits.size(1):
                target_idx = torch.clamp(target_idx, 0, logits.size(1) - 1)
            
            # InfoNCE computation
            pos_sim = torch.gather(similarities, 1, target_idx)
            neg_sim_sum = similarities.sum(dim=1, keepdim=True)
            info_nce_loss = -torch.log(pos_sim / neg_sim_sum)
            cat_loss += info_nce_loss.mean()
        
        return cat_loss

    def compute_contrastive_loss(self, proj):
        """Compute global contrastive loss"""
        sim_matrix = torch.matmul(proj, proj.T) / self.temperature
        labels = torch.arange(proj.size(0), device=sim_matrix.device)
        return F.cross_entropy(sim_matrix, labels)

    def compute_total_loss(self, num_features, cat_features, model_outputs):
        """Compute combined loss with all components"""
        losses = {}
        
        # Individual losses
        losses['num_loss'] = self.compute_numeric_loss(
            model_outputs['num_preds'], 
            num_features
        )
        
        losses['cat_loss'] = self.compute_categorical_loss(
            model_outputs['cat_logits'], 
            cat_features
        )
        
        losses['contrastive_loss'] = self.compute_contrastive_loss(
            model_outputs['proj']
        )
        
        # Combined loss with weights
        losses['total_loss'] = (
            0.5 * losses['num_loss'] +
            0.3 * losses['cat_loss'] +
            0.2 * losses['contrastive_loss']
        )
        
        return losses





class PretrainingDataset(Dataset):
    def __init__(self, num_features: List[torch.Tensor],
                 cat_features: List[torch.Tensor],
                 cat_feature_info: Dict,
                 mask_ratio: float = 0.45):
        """
        Dataset for self-supervised pretraining with masked feature prediction.

        Args:
            num_features: List of numerical feature tensors
            cat_features: List of categorical feature tensors
            cat_feature_info: Dictionary containing categorical feature information
            mask_ratio: Fraction of features to mask for prediction
        """
        self.num_features = num_features
        self.cat_features = cat_features
        self.cat_feature_info = cat_feature_info
        self.mask_ratio = mask_ratio

        # Verify all features have same length
        lengths = ([f.shape[0] for f in num_features] +
                   [f.shape[0] for f in cat_features])
        assert len(set(lengths)) == 1, "All features must have same length"

        self.dataset_length = lengths[0]
        self.num_dims = [f.shape[1] for f in num_features]
        self.cat_dims = [1 for _ in cat_features]

        # Calculate mask token indices for categorical features
        self.mask_tokens = [info["categories"] for info in cat_feature_info.values()]

        # Pre-generate masks for the entire dataset
        self.input_features, self.targets = self.generate_masks()

    def __len__(self):
        return self.dataset_length

    def generate_masks(self):
        """
        Generate masks for the entire dataset at once.
        Returns:
            Tuple of (masked_features, targets, masks)
        """
        # Clone features for masked version
        input_num_features  = [f.clone() for f in self.num_features]
        input_cat_features = [f.clone() for f in self.cat_features]

        # Create targets
        num_targets = [f.clone() for f in self.num_features]
        cat_targets = [f.clone() for f in self.cat_features]


        return (input_num_features,  input_cat_features), (num_targets, cat_targets)

    def __getitem__(self, idx):
        """
        Get a batch of masked and target features.
        """
        # Get masked features
        input_num_features = [f[idx] for f in self.input_features[0]]
        input_cat_features = [f[idx] for f in self.input_features[1]]

        # Get targets
        target_num = [f[idx] for f in self.targets[0]]
        target_cat = [f[idx] for f in self.targets[1]]

        # Get masks
        # num_masks = [m[idx] for m in self.masks[0]]
        # cat_masks = [m[idx] for m in self.masks[1]]

        # Move tensors to device
        input_num_features = [f for f in input_num_features]
        input_cat_features = [f for f in input_cat_features]
        target_num = [f for f in target_num]
        target_cat = [f for f in target_cat]
        # num_masks = [m for m in num_masks]
        # cat_masks = [m for m in cat_masks]

        return (input_num_features, input_cat_features), (target_num, target_cat)



def pretrain_model(model, train_loader, val_loader, num_epochs, device, lr=0.001):
    """Pretrain the model using masked feature prediction."""
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # num_criterion = nn.MSELoss(reduction='none')
    # cat_criterion = nn.CrossEntropyLoss(reduction='none')

    print("\nStarting Pretraining:")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (input_features, targets) in enumerate(train_loader):
            input_num_features, input_cat_features = input_features
            num_features, cat_features = targets
            # num_masks, cat_masks = masks

            # Move to device
            input_num_features = [f.to(device) for f in input_num_features]
            input_cat_features = [f.to(device) for f in input_cat_features]
            num_features = [t.to(device) for t in num_features]
            cat_features = [t.to(device) for t in cat_features]

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_num_features, input_cat_features)

            loss = model.compute_total_loss(num_features, cat_features, outputs)
            loss = loss['total_loss']
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            # if batch_idx % 10 == 0:
            #     print(
            #         f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        # Validation step
        model.eval()
        val_loss = 0
        val_batches = 0
        
        if epoch % 5 == 0:
            with torch.no_grad():
                for batch_idx, (input_features, targets) in enumerate(val_loader):
                    input_num_features, input_cat_features = input_features
                    num_features, cat_features = targets
                    # num_masks, cat_masks = masks

                    # Move to device
                    input_num_features = [f.to(device) for f in input_num_features]
                    input_cat_features = [f.to(device) for f in input_cat_features]
                    num_features = [t.to(device) for t in num_features]
                    cat_features = [t.to(device) for t in cat_features]
                    # num_masks = [m.to(device) for m in num_masks]
                    # cat_masks = [m.to(device) for m in cat_masks]

                    # Forward pass
                    outputs = model(input_num_features, input_cat_features)

                    val_loss_batch = model.compute_total_loss(num_features, cat_features, outputs)
                    val_loss_batch = val_loss_batch['total_loss']

                    val_loss += val_loss_batch.item()
                    val_batches += 1

            avg_val_loss = val_loss / val_batches
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

    return model



# import math
# from collections import defaultdict

# class PretrainingModel(nn.Module):
#     def __init__(
#             self,
#             cat_feature_info: Dict,
#             num_feature_info: List,
#             model: nn.Module,
#             output_dim: int,
#             temperature: float = 0.07,  # Lower temperature for sharper contrasts
#             dropout: float = 0.2,      # Increased dropout
#             hidden_dim: int = 1024,    # Much larger hidden dimension
#             projection_dim: int = 512,  # Separate dimension for projection head
#             **kwargs,
#     ):
#         super().__init__()
        
#         self.output_dim = output_dim
#         self.temperature = temperature
#         self.encoder = model(cat_feature_info, num_feature_info, output_dim, config=kwargs.get('config'))
        
#         # Improved projection head with larger capacity and bottleneck
#         self.projector = nn.Sequential(
#             nn.Linear(output_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, hidden_dim),  # One more layer for deeper representation
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, projection_dim)  # Project to smaller dim for contrastive learning
#         )
        
#         # Improved numerical prediction heads
#         self.num_heads = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(output_dim, hidden_dim),
#                 nn.LayerNorm(hidden_dim),
#                 nn.GELU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(hidden_dim, hidden_dim // 2),
#                 nn.LayerNorm(hidden_dim // 2),
#                 nn.GELU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(hidden_dim // 2, 2)  # Predict both mean and variance
#             ) for _ in num_feature_info
#         ])
        
#         # Improved categorical prediction heads
#         self.cat_heads = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(output_dim, hidden_dim),
#                 nn.LayerNorm(hidden_dim),
#                 nn.GELU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(hidden_dim, info["categories"])
#             ) for info in cat_feature_info.values()
#         ])

#     def forward(self, num_features: List[torch.Tensor], cat_features: List[torch.Tensor]):
#         # Get base encoding
#         encoded = self.encoder(num_features, cat_features)
        
#         # Apply stochastic depth during training
#         if self.training:
#             encoded = F.dropout2d(encoded.unsqueeze(-1), p=0.1, training=True).squeeze(-1)
        
#         # Project for contrastive learning
#         proj = self.projector(encoded)
        
#         # Get predictions
#         num_preds = [head(encoded) for head in self.num_heads]
#         cat_logits = [head(encoded) for head in self.cat_heads]
        
#         return {
#             'encoded': encoded,
#             'proj': proj,
#             'num_preds': num_preds,
#             'cat_logits': cat_logits
#         }

#     def compute_losses(self, 
#                       num_features: List[torch.Tensor],
#                       cat_features: List[torch.Tensor],
#                       outputs: Dict,
#                       num_neg_samples: int = 50) -> Dict:
#         """Compute improved losses with better weighting and sampling"""
#         batch_size = num_features[0].size(0)
        
#         # Improved numerical loss using negative log likelihood with predicted variance
#         num_recon_loss = 0.0
#         for pred, target in zip(outputs['num_preds'], num_features):
#             mean, log_var = pred.chunk(2, dim=-1)
#             var = torch.exp(log_var)
#             num_recon_loss += torch.mean(0.5 * (
#                 torch.log(var) + 
#                 (target.view(-1, 1) - mean)**2 / var
#             ))
        
#         # # Improved categorical loss with hard negative mining
#         cat_loss = 0.0x
#         for logits, target in zip(outputs['cat_logits'], cat_features):
#             # Sample hard negatives
#             with torch.no_grad():
#                 similarities = F.softmax(logits, dim=1)
#                 pos_sim = torch.gather(similarities, 1, target.view(-1, 1))
#                 # Get number of classes for this categorical feature
#                 num_classes = logits.size(1)
#                 # Limit number of negative samples to available classes minus 1 (excluding positive)
#                 actual_neg_samples = min(num_neg_samples, num_classes - 1)
                
#                 # Create mask for all classes except the target
#                 negative_mask = torch.ones_like(similarities, dtype=torch.bool)
#                 negative_mask.scatter_(1, target.view(-1, 1), False)
                
#                 # Get top-k hardest negatives from non-target classes
#                 neg_sim = similarities.masked_fill(~negative_mask, float('-inf'))
#                 _, hard_negative_indices = neg_sim.topk(k=actual_neg_samples, dim=1)
                
#             # Compute weighted NCE loss
#             pos_logits = torch.gather(logits, 1, target.view(-1, 1))
#             neg_logits = torch.gather(logits, 1, hard_negative_indices)
            
#             cat_loss += F.cross_entropy(
#                 torch.cat([pos_logits, neg_logits], dim=1),
#                 torch.zeros(batch_size, device=logits.device).long()
#             )
        
#         # Improved contrastive loss with momentum features
#         normalized_proj = F.normalize(outputs['proj'], dim=1)
#         sim_matrix = torch.matmul(normalized_proj, normalized_proj.T)
#         sim_matrix = sim_matrix / self.temperature
        
#         # Use sharpened similarity targets
#         labels = torch.arange(batch_size, device=sim_matrix.device)
#         contrastive_loss = F.cross_entropy(sim_matrix, labels)
                
#         # Weight the losses
#         total_loss = (
#             0.5 * num_recon_loss + 
#             0.3 * cat_loss + 
#             0.2 * contrastive_loss
#         )
        
#         return {
#             'num_reconstruction_loss': num_recon_loss,
#             'cat_loss': cat_loss,
#             'contrastive_loss': contrastive_loss,
#             'total_loss': total_loss
#         }

# def pretrain_model(model, train_loader, val_loader, num_epochs, device, 
#                           lr=1e-3, warmup_epochs=5):
#     """Train with cosine learning rate schedule and warmup"""
#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    
#     # Cosine learning rate schedule with warmup
#     def get_lr_scale(epoch):
#         if epoch < warmup_epochs:
#             return 1
#         return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    
#     for epoch in range(num_epochs):
#         # Update learning rate
#         lr_scale = get_lr_scale(epoch)
#         for param_group in optimizer.param_groups:
#             optimizer.param_group['lr'] = lr * lr_scale
        
#         model.train()
#         total_loss = 0
#         num_batches = 0
        
#         for batch_num_features, batch_cat_features, _ in train_loader:
#             batch_num_features = [f.to(device) for f in batch_num_features]
#             batch_cat_features = [f.to(device) for f in batch_cat_features]
            
#             optimizer.zero_grad()
            
#             outputs = model(batch_num_features, batch_cat_features)
#             losses = model.compute_losses(batch_num_features, batch_cat_features, outputs)
            
#             loss = losses['total_loss']
#             loss.backward()
            
#             # Gradient clipping
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
#             optimizer.step()
            
#             total_loss += loss.item()
#             num_batches += 1
        
#         avg_loss = total_loss / num_batches
#         print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
#         # Validation with detailed metrics
#         if epoch % 5 == 0:
#             model.eval()
#             val_losses = defaultdict(float)
#             val_batches = 0
            
#             with torch.no_grad():
#                 for batch_num_features, batch_cat_features, _ in val_loader:
#                     batch_num_features = [f.to(device) for f in batch_num_features]
#                     batch_cat_features = [f.to(device) for f in batch_cat_features]
                    
#                     outputs = model(batch_num_features, batch_cat_features)
#                     losses = model.compute_losses(batch_num_features, batch_cat_features, outputs)
                    
#                     for k, v in losses.items():
#                         val_losses[k] += v.item()
#                     val_batches += 1
            
#             # Print detailed validation metrics
#             print("\nValidation Metrics:")
#             for k, v in val_losses.items():
#                 print(f"{k}: {v/val_batches:.4f}")
#             print()
    
#     return model

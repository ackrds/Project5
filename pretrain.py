from typing import List, Dict
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from mambular.base_models import BaseModel
from mambular.configs import DefaultMambularConfig

# class PretrainingModel(BaseModel):
#     def __init__(
#             self,
#             cat_feature_info,
#             num_feature_info,
#             model,
#             output_dim,
#             config=None,
#             **kwargs,
#     ):
#         super().__init__()
#         self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

#         # Base encoder with 64-dimensional output
#         self.output_dim = output_dim

#         self.encoder = model(cat_feature_info, num_feature_info, output_dim, config=config)

#         # Define prediction heads
#         self.num_heads = nn.ModuleList([
#             nn.Sequential(
#                 nn.SELU(),
#                 nn.Linear(output_dim, 16),
#                 nn.SELU(),
#                 nn.Linear(16, 1)
#             ) for feat in num_feature_info
#         ])

#         self.cat_heads = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(output_dim, 16),
#                 nn.ReLU(),
#                 nn.Linear(16, info["categories"])
#             ) for info in cat_feature_info.values()
#         ])

#     def forward(self, num_features, cat_features):

#         encoded = self.encoder(num_features, cat_features)

#         # Predict features
#         num_preds = [head(encoded) for head in self.num_heads]
#         cat_logits = [head(encoded) for head in self.cat_heads]

#         return num_preds, cat_logits


# class PretrainingDataset(Dataset):
#     def __init__(self, num_features: List[torch.Tensor],
#                  cat_features: List[torch.Tensor],
#                  cat_feature_info: Dict,
#                  mask_ratio: float = 0.15):
#         """
#         Dataset for self-supervised pretraining with masked feature prediction.

#         Args:
#             num_features: List of numerical feature tensors
#             cat_features: List of categorical feature tensors
#             cat_feature_info: Dictionary containing categorical feature information
#             mask_ratio: Fraction of features to mask for prediction
#         """
#         self.num_features = num_features
#         self.cat_features = cat_features
#         self.cat_feature_info = cat_feature_info
#         self.mask_ratio = mask_ratio

#         # Verify all features have same length
#         lengths = ([f.shape[0] for f in num_features] +
#                    [f.shape[0] for f in cat_features])
#         assert len(set(lengths)) == 1, "All features must have same length"

#         self.dataset_length = lengths[0]
#         self.num_dims = [f.shape[1] for f in num_features]
#         self.cat_dims = [1 for _ in cat_features]

#         # Calculate mask token indices for categorical features
#         self.mask_tokens = [info["categories"] for info in cat_feature_info.values()]

#         # Pre-generate masks for the entire dataset
#         self.masked_features, self.targets, self.masks = self.generate_masks()

#     def __len__(self):
#         return self.dataset_length

#     def generate_masks(self):
#         """
#         Generate masks for the entire dataset at once.
#         Returns:
#             Tuple of (masked_features, targets, masks)
#         """
#         # Clone features for masked version
#         masked_num = [f.clone() for f in self.num_features]
#         masked_cat = [f.clone() for f in self.cat_features]

#         # Create targets
#         num_targets = [f.clone() for f in self.num_features]
#         cat_targets = [f.clone() for f in self.cat_features]

#         # Generate masks for all features
#         num_masks = [torch.rand(f.shape) < self.mask_ratio for f in self.num_features]
#         cat_masks = [torch.rand(f.shape) < self.mask_ratio for f in self.cat_features]

#         # Apply masking to numerical features
#         for i, num_feat in enumerate(masked_num):
#             mask = num_masks[i]

#             if mask.any():
#                 # Get masked positions
#                 masked_indices = torch.where(mask)
#                 num_masked = len(masked_indices[0])
#                 shuffled_indices = torch.randperm(num_masked)

#                 # 50% replace with feature mean
#                 mask_80 = shuffled_indices[:int(0.8* num_masked)]
#                 if len(mask_80) > 0:
#                     rows_80, cols_80 = masked_indices[0][mask_80], masked_indices[1][mask_80]
#                     feat_means = num_feat.mean(dim=0)
#                     masked_num[i][rows_80, cols_80] = feat_means[cols_80]

#                 # 10% replace with random values
#                 mask_10 = shuffled_indices[int(0.8*num_masked):int(0.9 * num_masked)]
#                 if len(mask_10) > 0:
#                     rows_10, cols_10 = masked_indices[0][mask_10], masked_indices[1][mask_10]
#                     feat_stds = num_feat.std(dim=0)
#                     feat_means = num_feat.mean(dim=0)
#                     random_values = torch.randn(len(mask_10)) * feat_stds[cols_10] + feat_means[cols_10]
#                     masked_num[i][rows_10, cols_10] = random_values

#                 # 10% keep unchanged (already done by cloning)

#         # Apply masking to categorical features
#         for i, (cat_feat, num_categories) in enumerate(zip(masked_cat, self.mask_tokens)):
#             mask = cat_masks[i]

#             if mask.any():
#                 masked_indices = torch.where(mask)
#                 num_masked = len(masked_indices[0])
#                 shuffled_indices = torch.randperm(num_masked)

#                 # 80% replace with mask token (0)
#                 mask_80 = shuffled_indices[:int(0.8 * num_masked)]
#                 if len(mask_80) > 0:
#                     rows_80 = masked_indices[0][mask_80]
#                     masked_cat[i][rows_80] = 0

#                 # 10% replace with random categories
#                 mask_10 = shuffled_indices[int(0.8 * num_masked):int(0.9 * num_masked)]
#                 if len(mask_10) > 0:
#                     rows_10 = masked_indices[0][mask_10]
#                     random_cats = torch.randint(1, num_categories, (len(mask_10),))
#                     masked_cat[i][rows_10] = random_cats

#                 # 10% keep unchanged (already done by cloning)

#         return (masked_num, masked_cat), (num_targets, cat_targets), (num_masks, cat_masks)

#     def __getitem__(self, idx):
#         """
#         Get a batch of masked and target features.
#         """
#         # Get masked features
#         masked_num = [f[idx] for f in self.masked_features[0]]
#         masked_cat = [f[idx] for f in self.masked_features[1]]

#         # Get targets
#         target_num = [f[idx] for f in self.targets[0]]
#         target_cat = [f[idx] for f in self.targets[1]]

#         # Get masks
#         num_masks = [m[idx] for m in self.masks[0]]
#         cat_masks = [m[idx] for m in self.masks[1]]

#         # Move tensors to device
#         masked_num = [f for f in masked_num]
#         masked_cat = [f for f in masked_cat]
#         target_num = [f for f in target_num]
#         target_cat = [f for f in target_cat]
#         num_masks = [m for m in num_masks]
#         cat_masks = [m for m in cat_masks]

#         return (masked_num, masked_cat), (target_num, target_cat), (num_masks, cat_masks)



# def pretrain_model(model, train_loader, val_loader, num_epochs, device, lr=0.001):
#     """Pretrain the model using masked feature prediction."""
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     num_criterion = nn.MSELoss(reduction='none')
#     cat_criterion = nn.CrossEntropyLoss(reduction='none')

#     print("\nStarting Pretraining:")
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         num_batches = 0

#         for batch_idx, (masked_features, targets, masks) in enumerate(train_loader):
#             masked_num, masked_cat = masked_features
#             num_targets, cat_targets = targets
#             num_masks, cat_masks = masks

#             # Move to device
#             masked_num = [f.to(device) for f in masked_num]
#             masked_cat = [f.to(device) for f in masked_cat]
#             num_targets = [t.to(device) for t in num_targets]
#             cat_targets = [t.to(device) for t in cat_targets]
#             num_masks = [m.to(device) for m in num_masks]
#             cat_masks = [m.to(device) for m in cat_masks]

#             optimizer.zero_grad()

#             # Forward pass
#             num_preds, cat_logits = model(masked_num, masked_cat)

#             # Calculate losses
#             loss = 0

#             # Numerical feature losses
#             for pred, target, mask in zip(num_preds, num_targets, num_masks):
#                 if mask.any():
#                     num_loss = num_criterion(pred, target)
#                     loss += (num_loss * mask.float()).mean()

#             # Categorical feature losses
#             for logits, target, mask in zip(cat_logits, cat_targets, cat_masks):
#                 if mask.any():
#                     cat_loss = cat_criterion(
#                         logits.view(-1, logits.size(-1)),
#                         target.long().view(-1)
#                     )
#                     loss += (cat_loss * mask.view(-1).float()).mean()

#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             num_batches += 1

#             # if batch_idx % 10 == 0:
#             #     print(
#             #         f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

#         avg_loss = total_loss / num_batches
#         print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

#         # Validation step
#         model.eval()
#         val_loss = 0
#         val_batches = 0
        
#         if epoch % 5 == 0:
#             with torch.no_grad():
#                 for batch_idx, (masked_features, targets, masks) in enumerate(val_loader):
#                     masked_num, masked_cat = masked_features
#                     num_targets, cat_targets = targets
#                     num_masks, cat_masks = masks

#                     # Move to device
#                     masked_num = [f.to(device) for f in masked_num]
#                     masked_cat = [f.to(device) for f in masked_cat]
#                     num_targets = [t.to(device) for t in num_targets]
#                     cat_targets = [t.to(device) for t in cat_targets]
#                     num_masks = [m.to(device) for m in num_masks]
#                     cat_masks = [m.to(device) for m in cat_masks]

#                     # Forward pass
#                     num_preds, cat_logits = model(masked_num, masked_cat)

#                     # Calculate validation losses
#                     val_loss_batch = 0

#                     # Numerical feature losses
#                     for pred, target, mask in zip(num_preds, num_targets, num_masks):
#                         if mask.any():
#                             num_loss = num_criterion(pred, target)
#                             val_loss_batch += (num_loss * mask.float()).mean()

#                     # Categorical feature losses
#                     for logits, target, mask in zip(cat_logits, cat_targets, cat_masks):
#                         if mask.any():
#                             cat_loss = cat_criterion(
#                                 logits.view(-1, logits.size(-1)),
#                                 target.long().view(-1)
#                             )
#                             val_loss_batch += (cat_loss * mask.view(-1).float()).mean()

#                     val_loss += val_loss_batch.item()
#                     val_batches += 1

#             avg_val_loss = val_loss / val_batches
#             print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

#     return model


from typing import List, Dict, Tuple
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset


# class PretrainingModel(nn.Module):
#     def __init__(
#             self,
#             cat_feature_info: Dict,
#             num_feature_info: List,
#             model: nn.Module,
#             output_dim: int,
#             temperature: float = 0.2,
#             dropout: float = 0.1,
#             config=None,
#             **kwargs,
#     ):
#         super().__init__()
        
#         # Base encoder that handles embeddings internally
#         self.output_dim = output_dim
#         self.temperature = temperature
#         self.encoder = model(cat_feature_info, num_feature_info, output_dim, config=config)
        
#         # Projection head for contrastive learning
#         self.projector = nn.Sequential(
#             nn.Linear(output_dim, output_dim),
#             nn.ReLU(),
#             nn.Linear(output_dim, output_dim)
#         )
        
#         # Prediction heads for numerical features
#         self.num_heads = nn.ModuleList([
#             nn.Sequential(
#                 nn.BatchNorm1d(output_dim),
#                 nn.Dropout(dropout),
#                 nn.SELU(),
#                 nn.Linear(output_dim, int(output_dim/2)),
#                 nn.Dropout(dropout),
#                 nn.SELU(),
#                 nn.Linear(int(output_dim/2), 1)
#             ) for _ in num_feature_info
#         ])
        
#         # Prediction heads for categorical features
#         self.cat_heads = nn.ModuleList([
#             nn.Sequential(
#                 nn.BatchNorm1d(output_dim),
#                 nn.Dropout(dropout),
#                 nn.SELU(),
#                 nn.Linear(output_dim, int(output_dim/2)),
#                 nn.Dropout(dropout),
#                 nn.SELU(),
#                 nn.Linear(int(output_dim/2), info["categories"])
#             ) for info in cat_feature_info.values()
#         ])

#     def manifold_mixup(self, num_features: List[torch.Tensor], 
#                       cat_features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, float]:
#         """Apply manifold mixup to encoded representations"""
#         # Get base encoding
#         encoded = self.encoder(num_features, cat_features)
#         batch_size = encoded.size(0)
        
#         # Generate permutation for mixup
#         perm = torch.randperm(batch_size, device=encoded.device)
#         lam = 0.75  # As specified in the paper
        
#         # Apply mixup to encoded representations
#         mixed_encoded = lam * encoded + (1 - lam) * encoded[perm]
        
#         return mixed_encoded, perm, lam

#     def forward(self, num_features: List[torch.Tensor], 
#                 cat_features: List[torch.Tensor], 
#                 apply_mixup: bool = True):
#         # Get base encoding
#         encoded = self.encoder(num_features, cat_features)
#         proj = self.projector(encoded)
        
#         if apply_mixup:
#             # Apply manifold mixup
#             mixed_encoded, perm, lam = self.manifold_mixup(num_features, cat_features)
#             mixed_proj = self.projector(mixed_encoded)
            
#             # Get predictions for mixed encoding
#             mixed_num_preds = [head(mixed_encoded) for head in self.num_heads]
#             mixed_cat_logits = [head(mixed_encoded) for head in self.cat_heads]
#         else:
#             mixed_encoded = None
#             mixed_proj = None
#             perm = None
#             lam = None
#             mixed_num_preds = None
#             mixed_cat_logits = None
        
#         # Get predictions for original encoding
#         num_preds = [head(encoded) for head in self.num_heads]
#         cat_logits = [head(encoded) for head in self.cat_heads]
        
#         return {
#             'encoded': encoded,
#             'proj': proj,
#             'num_preds': num_preds,
#             'cat_logits': cat_logits,
#             'mixed_encoded': mixed_encoded,
#             'mixed_proj': mixed_proj,
#             'mixed_num_preds': mixed_num_preds,
#             'mixed_cat_logits': mixed_cat_logits,
#             'perm': perm,
#             'lam': lam
#         }

#     def compute_losses(self, 
#                       num_features: List[torch.Tensor],
#                       cat_features: List[torch.Tensor],
#                       outputs: Dict) -> Dict:
#         """Compute reconstruction and contrastive losses"""
#         batch_size = num_features[0].size(0)
#         recon_loss = 0.0
        
#         # MAE loss for numerical features
#         for pred, target in zip(outputs['num_preds'], num_features):
#             recon_loss += F.l1_loss(pred, target.view(-1, 1))
            
#         # NCE loss for categorical features
#         for logits, target in zip(outputs['cat_logits'], cat_features):
#             # Generate negative samples
#             num_classes = logits.size(1)
#             num_neg = 10
            
#             # Get positive logits
#             pos_logits = torch.gather(logits, 1, target.view(-1, 1))
            
#             # Generate negative samples avoiding the positive class
#             neg_samples = []
#             for i in range(batch_size):
#                 available_classes = list(range(num_classes))
#                 available_classes.remove(target[i].item())
#                 neg_idx = torch.tensor(available_classes)[torch.randperm(len(available_classes))[:num_neg]]
#                 neg_samples.append(neg_idx)
#             neg_samples = torch.stack(neg_samples).to(target.device)
            
#             # Get negative logits
#             neg_logits = torch.gather(logits, 1, neg_samples)
            
#             # Compute NCE loss
#             nce_loss = -torch.log(
#                 torch.exp(pos_logits) / 
#                 (torch.exp(pos_logits) + torch.sum(torch.exp(neg_logits), dim=1, keepdim=True))
#             ).mean()
            
#             recon_loss += nce_loss
            
#         # Contrastive loss if mixup was applied
#         contrastive_loss = 0.0
#         if outputs['mixed_proj'] is not None:
#             normalized_proj = F.normalize(outputs['proj'], dim=1)
#             normalized_mixed_proj = F.normalize(outputs['mixed_proj'], dim=1)
            
#             sim_matrix = torch.matmul(normalized_proj, normalized_mixed_proj.T) / self.temperature
#             labels = torch.arange(batch_size, device=sim_matrix.device)
#             contrastive_loss = F.cross_entropy(sim_matrix, labels)
        
#         return {
#             'reconstruction_loss': recon_loss,
#             'contrastive_loss': contrastive_loss,
#             'total_loss': recon_loss + contrastive_loss
#         }

class PretrainingDataset(Dataset):
    def __init__(self, num_features, cat_features):
        self.num_features = num_features
        self.cat_features = cat_features
        
        # Verify all features have same length
        lengths = ([f.shape[0] for f in num_features] +
                  [f.shape[0] for f in cat_features])
        assert len(set(lengths)) == 1, "All features must have same length"
        
        self.length = lengths[0]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        num_feats = [f[idx] for f in self.num_features]
        cat_feats = [f[idx] for f in self.cat_features]

        num_feats = [f.float() if not f.is_floating_point() else f for f in num_feats]
        cat_feats = [f.long() if not f.dtype == torch.long else f for f in cat_feats]
        return num_feats, cat_feats, idx


# def pretrain_model(model, train_loader, val_loader, num_epochs, device, lr=0.001):
#     """Train the SSL model"""
#     optimizer = optim.Adam(model.parameters(), lr=lr)
    
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         num_batches = 0
        
#         for batch_num_features, batch_cat_features, _ in train_loader:
#             # Move features to device
#             batch_num_features = [f.to(device) for f in batch_num_features]
#             batch_cat_features = [f.to(device) for f in batch_cat_features]
            
#             optimizer.zero_grad()
            
#             # Forward pass
#             outputs = model(batch_num_features, batch_cat_features)
#             losses = model.compute_losses(batch_num_features, batch_cat_features, outputs)
            
#             loss = losses['total_loss']
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
#             num_batches += 1
        
#         avg_loss = total_loss / num_batches
#         print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
#         # Validation
#         if epoch % 5 == 0:
#             model.eval()
#             val_loss = 0
#             val_batches = 0
            
#             with torch.no_grad():
#                 for batch_num_features, batch_cat_features, _ in val_loader:
#                     batch_num_features = [f.to(device) for f in batch_num_features]
#                     batch_cat_features = [f.to(device) for f in batch_cat_features]
                    
#                     outputs = model(batch_num_features, batch_cat_features)
#                     losses = model.compute_losses(batch_num_features, batch_cat_features, outputs)
#                     val_loss += losses['total_loss'].item()
#                     val_batches += 1
                    
#             avg_val_loss = val_loss / val_batches
#             print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
    
#     return model

import math
from collections import defaultdict

class PretrainingModel(nn.Module):
    def __init__(
            self,
            cat_feature_info: Dict,
            num_feature_info: List,
            model: nn.Module,
            output_dim: int,
            temperature: float = 0.07,  # Lower temperature for sharper contrasts
            dropout: float = 0.2,      # Increased dropout
            hidden_dim: int = 1024,    # Much larger hidden dimension
            projection_dim: int = 512,  # Separate dimension for projection head
            **kwargs,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.temperature = temperature
        self.encoder = model(cat_feature_info, num_feature_info, output_dim, config=kwargs.get('config'))
        
        # Improved projection head with larger capacity and bottleneck
        self.projector = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),  # One more layer for deeper representation
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim)  # Project to smaller dim for contrastive learning
        )
        
        # Improved numerical prediction heads
        self.num_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 2)  # Predict both mean and variance
            ) for _ in num_feature_info
        ])
        
        # Improved categorical prediction heads
        self.cat_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, info["categories"])
            ) for info in cat_feature_info.values()
        ])

    def forward(self, num_features: List[torch.Tensor], cat_features: List[torch.Tensor]):
        # Get base encoding
        encoded = self.encoder(num_features, cat_features)
        
        # Apply stochastic depth during training
        if self.training:
            encoded = F.dropout2d(encoded.unsqueeze(-1), p=0.1, training=True).squeeze(-1)
        
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

    def compute_losses(self, 
                      num_features: List[torch.Tensor],
                      cat_features: List[torch.Tensor],
                      outputs: Dict,
                      num_neg_samples: int = 50) -> Dict:
        """Compute improved losses with better weighting and sampling"""
        batch_size = num_features[0].size(0)
        
        # Improved numerical loss using negative log likelihood with predicted variance
        num_recon_loss = 0.0
        for pred, target in zip(outputs['num_preds'], num_features):
            mean, log_var = pred.chunk(2, dim=-1)
            var = torch.exp(log_var)
            num_recon_loss += torch.mean(0.5 * (
                torch.log(var) + 
                (target.view(-1, 1) - mean)**2 / var
            ))
        
        # Improved categorical loss with hard negative mining
        cat_loss = 0.0
        for logits, target in zip(outputs['cat_logits'], cat_features):
            # Sample hard negatives
            with torch.no_grad():
                similarities = F.softmax(logits, dim=1)
                pos_sim = torch.gather(similarities, 1, target.view(-1, 1))
                # Get number of classes for this categorical feature
                num_classes = logits.size(1)
                # Limit number of negative samples to available classes minus 1 (excluding positive)
                actual_neg_samples = min(num_neg_samples, num_classes - 1)
                
                # Create mask for all classes except the target
                negative_mask = torch.ones_like(similarities, dtype=torch.bool)
                negative_mask.scatter_(1, target.view(-1, 1), False)
                
                # Get top-k hardest negatives from non-target classes
                neg_sim = similarities.masked_fill(~negative_mask, float('-inf'))
                _, hard_negative_indices = neg_sim.topk(k=actual_neg_samples, dim=1)
                
            # Compute weighted NCE loss
            pos_logits = torch.gather(logits, 1, target.view(-1, 1))
            neg_logits = torch.gather(logits, 1, hard_negative_indices)
            
            cat_loss += F.cross_entropy(
                torch.cat([pos_logits, neg_logits], dim=1),
                torch.zeros(batch_size, device=logits.device).long()
            )
        
        # Improved contrastive loss with momentum features
        normalized_proj = F.normalize(outputs['proj'], dim=1)
        sim_matrix = torch.matmul(normalized_proj, normalized_proj.T)
        sim_matrix = sim_matrix / self.temperature
        
        # Use sharpened similarity targets
        labels = torch.arange(batch_size, device=sim_matrix.device)
        contrastive_loss = F.cross_entropy(sim_matrix, labels)
                
        # Weight the losses
        total_loss = (
            0.5 * num_recon_loss + 
            0.3 * cat_loss + 
            0.2 * contrastive_loss
        )
        
        return {
            'num_reconstruction_loss': num_recon_loss,
            'cat_loss': cat_loss,
            'contrastive_loss': contrastive_loss,
            'total_loss': total_loss
        }

def pretrain_model(model, train_loader, val_loader, num_epochs, device, 
                          lr=1e-3, warmup_epochs=5):
    """Train with cosine learning rate schedule and warmup"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    
    # Cosine learning rate schedule with warmup
    def get_lr_scale(epoch):
        if epoch < warmup_epochs:
            return 1
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    
    for epoch in range(num_epochs):
        # Update learning rate
        lr_scale = get_lr_scale(epoch)
        for param_group in optimizer.param_groups:
            optimizer.param_group['lr'] = lr * lr_scale
        
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_num_features, batch_cat_features, _ in train_loader:
            batch_num_features = [f.to(device) for f in batch_num_features]
            batch_cat_features = [f.to(device) for f in batch_cat_features]
            
            optimizer.zero_grad()
            
            outputs = model(batch_num_features, batch_cat_features)
            losses = model.compute_losses(batch_num_features, batch_cat_features, outputs)
            
            loss = losses['total_loss']
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Validation with detailed metrics
        if epoch % 5 == 0:
            model.eval()
            val_losses = defaultdict(float)
            val_batches = 0
            
            with torch.no_grad():
                for batch_num_features, batch_cat_features, _ in val_loader:
                    batch_num_features = [f.to(device) for f in batch_num_features]
                    batch_cat_features = [f.to(device) for f in batch_cat_features]
                    
                    outputs = model(batch_num_features, batch_cat_features)
                    losses = model.compute_losses(batch_num_features, batch_cat_features, outputs)
                    
                    for k, v in losses.items():
                        val_losses[k] += v.item()
                    val_batches += 1
            
            # Print detailed validation metrics
            print("\nValidation Metrics:")
            for k, v in val_losses.items():
                print(f"{k}: {v/val_batches:.4f}")
            print()
    
    return model

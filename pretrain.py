from typing import List, Dict
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from mambular.base_models import BaseModel
from mambular.configs import DefaultMambularConfig


class PretrainingDataset(Dataset):
    def __init__(self, num_features: List[torch.Tensor],
                 cat_features: List[torch.Tensor],
                 cat_feature_info: Dict,
                 mask_ratio: float = 0.15):
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
        self.masked_features, self.targets, self.masks = self.generate_masks()

    def __len__(self):
        return self.dataset_length

    def generate_masks(self):
        """
        Generate masks for the entire dataset at once.
        Returns:
            Tuple of (masked_features, targets, masks)
        """
        # Clone features for masked version
        masked_num = [f.clone() for f in self.num_features]
        masked_cat = [f.clone() for f in self.cat_features]

        # Create targets
        num_targets = [f.clone() for f in self.num_features]
        cat_targets = [f.clone() for f in self.cat_features]

        # Generate masks for all features
        num_masks = [torch.rand(f.shape) < self.mask_ratio for f in self.num_features]
        cat_masks = [torch.rand(f.shape) < self.mask_ratio for f in self.cat_features]

        # Apply masking to numerical features
        for i, num_feat in enumerate(masked_num):
            mask = num_masks[i]

            if mask.any():
                # Get masked positions
                masked_indices = torch.where(mask)
                num_masked = len(masked_indices[0])
                shuffled_indices = torch.randperm(num_masked)

                # 80% replace with feature mean
                mask_80 = shuffled_indices[:int(0.8 * num_masked)]
                if len(mask_80) > 0:
                    rows_80, cols_80 = masked_indices[0][mask_80], masked_indices[1][mask_80]
                    feat_means = num_feat.mean(dim=0)
                    masked_num[i][rows_80, cols_80] = feat_means[cols_80]

                # 10% replace with random values
                mask_10 = shuffled_indices[int(0.8 * num_masked):int(0.9 * num_masked)]
                if len(mask_10) > 0:
                    rows_10, cols_10 = masked_indices[0][mask_10], masked_indices[1][mask_10]
                    feat_stds = num_feat.std(dim=0)
                    feat_means = num_feat.mean(dim=0)
                    random_values = torch.randn(len(mask_10)) * feat_stds[cols_10] + feat_means[cols_10]
                    masked_num[i][rows_10, cols_10] = random_values

                # 10% keep unchanged (already done by cloning)

        # Apply masking to categorical features
        for i, (cat_feat, num_categories) in enumerate(zip(masked_cat, self.mask_tokens)):
            mask = cat_masks[i]

            if mask.any():
                masked_indices = torch.where(mask)
                num_masked = len(masked_indices[0])
                shuffled_indices = torch.randperm(num_masked)

                # 80% replace with mask token (0)
                mask_80 = shuffled_indices[:int(0.8 * num_masked)]
                if len(mask_80) > 0:
                    rows_80 = masked_indices[0][mask_80]
                    masked_cat[i][rows_80] = 0

                # 10% replace with random categories
                mask_10 = shuffled_indices[int(0.8 * num_masked):int(0.9 * num_masked)]
                if len(mask_10) > 0:
                    rows_10 = masked_indices[0][mask_10]
                    random_cats = torch.randint(1, num_categories, (len(mask_10),))
                    masked_cat[i][rows_10] = random_cats

                # 10% keep unchanged (already done by cloning)

        return (masked_num, masked_cat), (num_targets, cat_targets), (num_masks, cat_masks)

    def __getitem__(self, idx):
        """
        Get a batch of masked and target features.
        """
        # Get masked features
        masked_num = [f[idx] for f in self.masked_features[0]]
        masked_cat = [f[idx] for f in self.masked_features[1]]

        # Get targets
        target_num = [f[idx] for f in self.targets[0]]
        target_cat = [f[idx] for f in self.targets[1]]

        # Get masks
        num_masks = [m[idx] for m in self.masks[0]]
        cat_masks = [m[idx] for m in self.masks[1]]

        return (masked_num, masked_cat), (target_num, target_cat), (num_masks, cat_masks)

class PretrainingModel(BaseModel):
    def __init__(
            self,
            cat_feature_info,
            num_feature_info,
            model,
            config=None,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        # Base encoder with 64-dimensional output
        self.output_dim = 32

        if config is not None:
            self.encoder = model(cat_feature_info, num_feature_info, 32, config=config)
        else:
            config = DefaultMambularConfig()
            # config = {*config}
            # config.d_model = 32
            self.encoder = model(cat_feature_info, num_feature_info, 32, config=config)


        self.encoder = model(cat_feature_info, num_feature_info, 32)
        self.encoder.hparams.use_embeddings = True



        # Define prediction heads
        self.num_heads = nn.ModuleList([
            nn.Sequential(
                nn.SELU(),
                nn.Linear(32, 32),
                nn.SELU(),
                nn.Linear(32, 1)
            ) for feat in num_feature_info
        ])

        self.cat_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, info["categories"])
            ) for info in cat_feature_info.values()
        ])

    def forward(self, num_features, cat_features):

        encoded = self.encoder(num_features, cat_features)

        # Predict features
        num_preds = [head(encoded) for head in self.num_heads]
        cat_logits = [head(encoded) for head in self.cat_heads]

        return num_preds, cat_logits


def pretrain_model(model, train_loader, num_epochs, device, lr=0.001):
    """Pretrain the model using masked feature prediction."""
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_criterion = nn.MSELoss(reduction='none')
    cat_criterion = nn.CrossEntropyLoss(reduction='none')

    print("\nStarting Pretraining:")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (masked_features, targets, masks) in enumerate(train_loader):
            masked_num, masked_cat = masked_features
            num_targets, cat_targets = targets
            num_masks, cat_masks = masks

            # Move to device
            masked_num = [f.to(device) for f in masked_num]
            masked_cat = [f.to(device) for f in masked_cat]
            num_targets = [t.to(device) for t in num_targets]
            cat_targets = [t.to(device) for t in cat_targets]
            num_masks = [m.to(device) for m in num_masks]
            cat_masks = [m.to(device) for m in cat_masks]

            optimizer.zero_grad()

            # Forward pass
            num_preds, cat_logits = model(masked_num, masked_cat)

            # Calculate losses
            loss = 0

            # Numerical feature losses
            for pred, target, mask in zip(num_preds, num_targets, num_masks):
                if mask.any():
                    num_loss = num_criterion(pred, target)
                    loss += (num_loss * mask.float()).mean()

            # Categorical feature losses
            for logits, target, mask in zip(cat_logits, cat_targets, cat_masks):
                if mask.any():
                    cat_loss = cat_criterion(
                        logits.view(-1, logits.size(-1)),
                        target.long().view(-1)
                    )
                    loss += (cat_loss * mask.view(-1).float()).mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    return model
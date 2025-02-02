import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.preprocessing import StandardScaler
import numpy as np
from mambular.base_models import  * 
from mambular.arch_utils.neural_decision_tree import NeuralDecisionTree
from mambular.configs.ndtf_config import DefaultNDTFConfig
from mambular.utils.get_feature_dimensions import get_feature_dimensions
from mambular.base_models import BaseModel

from saint.model import SAINT
from mambular.configs import *
from preprocessing import hash_features, split_df


class NDTF(BaseModel):
    def __init__(
        self,
        cat_feature_info,
        num_feature_info,
        num_classes: int = 2,
        config: DefaultNDTFConfig = DefaultNDTFConfig(),
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])

        self.cat_feature_info = cat_feature_info
        self.num_feature_info = num_feature_info
        self.num_classes = num_classes
        self.returns_ensemble = False

        input_dim = get_feature_dimensions(num_feature_info, cat_feature_info)

        self.input_dimensions = [input_dim]
        for _ in range(self.hparams.n_ensembles - 1):
            self.input_dimensions.append(np.random.randint(1, input_dim))

        # Initialize trees with correct output dimension (num_classes)
        self.trees = nn.ModuleList(
            [
                NeuralDecisionTree(
                    input_dim=self.input_dimensions[idx],
                    depth=np.random.randint(self.hparams.min_depth, self.hparams.max_depth),
                    output_dim=num_classes,  # Set to num_classes for classification
                    lamda=self.hparams.lamda,
                    temperature=self.hparams.temperature + np.abs(np.random.normal(0, 0.1)),
                    node_sampling=self.hparams.node_sampling,
                )
                for idx in range(self.hparams.n_ensembles)
            ]
        )

        self.conv_layer = nn.Conv1d(
            in_channels=self.input_dimensions[0],
            out_channels=1,
            kernel_size=self.input_dimensions[0],
            padding=self.input_dimensions[0] - 1,
            bias=True,
        )

        # Modified tree weights to match output dimensions
        self.tree_weights = nn.Parameter(
            torch.full((self.hparams.n_ensembles, num_classes), 1.0 / self.hparams.n_ensembles),
            requires_grad=True,
        )

    def forward(self, num_features, cat_features) -> torch.Tensor:
        x = num_features + cat_features
        x = torch.cat(x, dim=1)
        x = self.conv_layer(x.unsqueeze(2))
        x = x.transpose(1, 2).squeeze(-1)

        preds = []
        for idx, tree in enumerate(self.trees):
            tree_input = x[:, :self.input_dimensions[idx]]
            preds.append(tree(tree_input, return_penalty=False))

        # Stack predictions and handle dimensions
        preds = torch.stack(preds, dim=1)  # Shape: [batch_size, n_ensembles, num_classes]
        
        # Weighted sum across trees
        weighted_preds = torch.einsum('bec,ec->bc', preds, self.tree_weights)
        return weighted_preds

    def penalty_forward(self, num_features, cat_features) -> tuple[torch.Tensor, torch.Tensor]:
        x = num_features + cat_features
        x = torch.cat(x, dim=1)
        x = self.conv_layer(x.unsqueeze(2))
        x = x.transpose(1, 2).squeeze(-1)

        penalty = 0.0
        preds = []

        for idx, tree in enumerate(self.trees):
            tree_input = x[:, :self.input_dimensions[idx]]
            pred, pen = tree(tree_input, return_penalty=True)
            preds.append(pred)
            penalty += pen

        # Stack predictions and handle dimensions
        preds = torch.stack(preds, dim=1)  # Shape: [batch_size, n_ensembles, num_classes]
        
        # Weighted sum across trees
        weighted_preds = torch.einsum('bec,ec->bc', preds, self.tree_weights)
        return weighted_preds, self.hparams.penalty_factor * penalty

class MainDataset(Dataset):
    """Dataset class compatible with the provided training framework"""
    def __init__(self, num_features, cat_features, targets):
        self.num_features = num_features
        self.cat_features = cat_features
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        num_feat = [f[idx] for f in self.num_features]
        cat_feat = [f[idx] for f in self.cat_features]
        return num_feat, cat_feat, self.targets[idx]

def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Parameters from arguments
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    weight_decay = args.weight_decay
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data preparation
    x_train_num, x_train_cat, x_val_num, x_val_cat, x_test_num, x_test_cat, y_train, y_val, y_test, num_feature_info, cat_feature_info, test_columns = split_df(
        year=args.year, 
        month=args.month
    )

    # Scale numerical features
    scaler = StandardScaler()
    x_train_num_scaled = []
    x_val_num_scaled = []
    x_test_num_scaled = []
    
    for train_feat, val_feat, test_feat in zip(x_train_num, x_val_num, x_test_num):
        x_train_num_scaled.append(torch.tensor(scaler.fit_transform(train_feat), dtype=torch.float32).squeeze())
        x_val_num_scaled.append(torch.tensor(scaler.transform(val_feat), dtype=torch.float32).squeeze())
        x_test_num_scaled.append(torch.tensor(scaler.transform(test_feat), dtype=torch.float32).squeeze())

    # Convert categorical features to tensors
    x_train_cat = [torch.tensor(f, dtype=torch.long).squeeze()  for f in x_train_cat]
    x_val_cat = [torch.tensor(f, dtype=torch.long).squeeze()  for f in x_val_cat]
    x_test_cat = [torch.tensor(f, dtype=torch.long).squeeze()  for f in x_test_cat]

    # Create datasets
    train_dataset = MainDataset(x_train_num_scaled, x_train_cat, y_train)
    val_dataset = MainDataset(x_val_num_scaled, x_val_cat, y_val)
    test_dataset = MainDataset(x_test_num_scaled, x_test_cat, y_test)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    config = DefaultNDTFConfig()
    if args.config_values:
        for key, value in args.config_values.items():
            setattr(config, key, value)

    model = NDTF(
        cat_feature_info=cat_feature_info,
        num_feature_info=num_feature_info,
        num_classes=2,  # Binary classification
        config=config
    ).to(device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.t_max,
        eta_min=args.eta_min
    )

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for num_features, cat_features, targets in train_loader:
            # Move data to device
            num_features = [f.to(device) for f in num_features]
            cat_features = [f.to(device) for f in cat_features]
            targets = targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs, penalty = model.penalty_forward(num_features, cat_features)
            loss = criterion(outputs, targets) + penalty

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for num_features, cat_features, targets in val_loader:
                num_features = [f.to(device) for f in num_features]
                cat_features = [f.to(device) for f in cat_features]
                targets = targets.to(device)

                outputs, penalty = model.penalty_forward(num_features, cat_features)
                loss = criterion(outputs, targets) + penalty

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        # Update learning rate
        scheduler.step()

        # Print statistics
        if args.verbose:
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Train Acc: {100.*train_correct/train_total:.2f}%')
            print(f'Val Loss: {val_loss/len(val_loader):.4f}, '
                  f'Val Acc: {100.*val_correct/val_total:.2f}%')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    test_predictions = []

    model.eval()
    with torch.no_grad():
        for num_features, cat_features, targets in test_loader:
            num_features = [f.to(device) for f in num_features]
            cat_features = [f.to(device) for f in cat_features]
            targets = targets.to(device)

            outputs = model(num_features, cat_features)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            test_predictions.extend(predicted.cpu().numpy())

    print(f"Test Loss: {test_loss/len(test_loader):.4f}, "
          f"Test Accuracy: {100.*test_correct/test_total:.2f}%")
    
    # Calculate multipliers if needed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--t_max", type=int, default=40)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--ce_weight", type=float, default=0.5)
    parser.add_argument("--sce_weight", type=float, default=1.0)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--config_values", type=dict, default={})
    
    args = parser.parse_args()
    main(args)
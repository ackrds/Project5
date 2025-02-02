import torch
import numpy as np
from mambular.base_models import BaseModel
from torch.utils.data import Dataset, DataLoader
from torch import nn

class Model(BaseModel):
    def __init__(
            self,
            cat_feature_info,
            num_feature_info,
            model,
            output_dim,
            pretrained_state_dict=None,
            config=None,
            hidden_dim=1024,  # Match pretraining hidden dim
            dropout=0.3,      # Slightly higher dropout for finetuning
            **kwargs,
    ):
        super().__init__()
        
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])
        self.model = model(cat_feature_info, num_feature_info, output_dim, config)
        self.output_dim = output_dim

        if pretrained_state_dict is not None:
            # Load pretrained weights with proper error handling
            try:
                missing_keys, unexpected_keys = self.model.load_state_dict(pretrained_state_dict, strict=False)
                print(f"Loaded pretrained weights. Missing keys: {missing_keys}")
                print(f"Unexpected keys: {unexpected_keys}")
                # Gradually unfreeze layers during training
                self.frozen = True
                for param in self.model.parameters():
                    param.requires_grad = False
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")

        # Improved output head architecture
        self.output_head = nn.Sequential(
            # First block with residual connection
            nn.Sequential(
                nn.LayerNorm(output_dim),
                nn.Linear(output_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, output_dim),
                nn.Dropout(dropout)
            ),
            # Add residual connection
            LambdaLayer(lambda x, y: x + y),
            
            # Second block
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
    def unfreeze_encoder(self):
        """Gradually unfreeze the encoder"""
        if self.frozen:
            print("Unfreezing encoder layers...")
            for param in self.model.parameters():
                param.requires_grad = True
            self.frozen = False

    def forward(self, num_features, cat_features):
        # Get encoded representations
        encoded = self.model(num_features, cat_features)
        
        # Apply output head with residual connections
        x = encoded
        for layer in self.output_head:
            if isinstance(layer, LambdaLayer):
                x = layer(x, encoded)  # Pass both current and residual
            else:
                x = layer(x)
        
        return torch.softmax(x, dim=-1)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
    
    def forward(self, x, *args):
        return self.lambd(x, *args)

class MainDataset(Dataset):
    def __init__(self, num_features, cat_features, labels):
        """
        Initialize training dataset for supervised learning.

        Args:
            num_features (list): List of numerical feature tensors
            cat_features (list): List of categorical feature tensors
            labels (tensor): Target labels
        """
        self.num_features = num_features
        self.cat_features = cat_features
        self.labels = labels

        # Validate inputs
        if not num_features and not cat_features:
            raise ValueError("At least one of num_features or cat_features must be provided")

        # Get dataset size from first feature
        self.length = len(num_features[0] if num_features else cat_features[0])

        # Validate all features have same length
        for feat in num_features + cat_features:
            if len(feat) != self.length:
                raise ValueError("All features must have the same length")

        if len(labels) != self.length:
            raise ValueError("Number of labels must match number of samples")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Get a single training example.

        Args:
            idx (int): Index of the example to fetch

        Returns:
            tuple: (numerical_features, categorical_features, label)
        """
        # Get numerical features
        num_feats = [feat[idx] for feat in self.num_features]

        # Get categorical features
        cat_feats = [feat[idx] for feat in self.cat_features]

        # Get label
        label = self.labels[idx]

        return num_feats, cat_feats, label

def evaluate_model(model, data_loader, criterion, device):
    """Evaluate the model on given data loader."""
    model.eval()
    total_loss = 0
    total_samples = 0
    total_correct = 0
    preds = []
    with torch.no_grad():
        for num_features, cat_features, labels in data_loader:
            # Move data to device
            num_features = [f.to(device) for f in num_features]
            cat_features = [f.to(device) for f in cat_features]

            labels = labels.to(device)
            labels = labels.squeeze()

            outputs = model(num_features, cat_features)
            loss = criterion(outputs, labels.long())

            # total_loss += loss.item() * labels.size(0)

            total_loss += loss.item()
            probabilities = outputs[:, 1]
            _, predicted = outputs.max(1)
            
            total_samples += labels.size(0)
            total_correct += predicted.eq(labels).sum().item()
            preds.append(probabilities)

    epoch_loss = total_loss / len(data_loader)
    epoch_acc = 100. * total_correct / total_samples
    preds = torch.cat(preds)
    # Store metrics
    return epoch_loss, epoch_acc, preds


# def test_model(model, test_data, criterion, device):
#     """Evaluate the model on given data loader."""
#     model.eval()
#     total_loss = 0
#     total_samples = 0
#     total_correct = 0
#     preds = []
#     with torch.no_grad():
#         for num_features, cat_features, labels in test_data:
#             # Move data to device
#             num_features = [f.to(device) for f in num_features]
#             cat_features = [f.to(device) for f in cat_features]

#             labels = labels.to(device)
#             labels = labels.squeeze()

#             outputs = model(num_features, cat_features)
#             loss = criterion(outputs, labels.long())

#             # total_loss += loss.item() * labels.size(0)

#             total_loss += loss.item()
#             probabilities = outputs[:, 1]
#             _, predicted = outputs.max(1)
            
#             total_samples += labels.size(0)
#             total_correct += predicted.eq(labels).sum().item()
#             preds.append(probabilities)

#     epoch_loss = total_loss / len(data_loader)
#     epoch_acc = 100. * total_correct / total_samples
#     preds = torch.cat(preds)
#     # Store metrics
#     return epoch_loss, epoch_acc, preds



def train_model(model, train_loader, val_loader, criterion, 
                                      optimizer, num_epochs, device, scheduler, verbose=0):
    """Enhanced training with gradual unfreezing and monitoring"""
    model.train()
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    # unfreeze_epoch = num_epochs // 3  # Unfreeze after 1/3 of training
    
    for epoch in range(num_epochs):
        # Gradual unfreezing
        # if epoch == unfreeze_epoch:
        #     model.unfreeze_encoder()
        #     # Reduce learning rate when unfreezing
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = param_group['lr'] * 0.1
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (num_feats, cat_feats, labels) in enumerate(train_loader):
            num_feats = [feat.to(device) for feat in num_feats]
            cat_feats = [feat.to(device) for feat in cat_feats]
            labels = labels.to(device).squeeze()
            
            optimizer.zero_grad()
            
            outputs = model(num_feats, cat_feats)
            loss = criterion(outputs, labels.long())
            
            # Add L2 regularization for unfrozen layers
            l2_lambda = 0.01
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                if param.requires_grad:
                    l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        # Validation
        val_loss, val_acc, _ = evaluate_model(model, val_loader, criterion, device)
        
        # Store metrics
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        # if epoch > unfreeze_epoch:
        scheduler.step()
        # else:
            # scheduler.step()
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
        
        if verbose or epoch % 5 == 0:
            print(f'Epoch {epoch}/{num_epochs}:')
            # print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'LR: {scheduler.optimizer.param_groups[0]["lr"]:.8f}')
            
    return model, history




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
            config = None,
            dropout=0.1,
            **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])
        self.model = model(cat_feature_info, num_feature_info, output_dim, config)
        self.output_dim = output_dim

        if pretrained_state_dict is not None:
            self.model.embedding_layer.load_state_dict(pretrained_state_dict)  # Fixed loading state dict
            print("Loaded pretrained state dict")
            # Freeze embedding layer
            for param in self.model.embedding_layer.parameters():
                param.requires_grad = False

        if self.output_dim > 2:
            self.output_head = nn.Sequential(
                nn.BatchNorm1d(output_dim),
                nn.Dropout(dropout),
                nn.SELU(),
                nn.Linear(output_dim, int(output_dim/2)),
                nn.BatchNorm1d(int(output_dim/2)),
                nn.Dropout(dropout),
                nn.SELU(),
                nn.Linear(int(output_dim/2), 2),
                nn.Sigmoid()
            )
        else:
            self.output_head = nn.Sequential(
                nn.BatchNorm1d(output_dim),
                nn.SELU(),
                nn.Linear(output_dim, 2),
                nn.Sigmoid()
            )

    def forward(self, num_features, cat_features):
        x = self.model(num_features, cat_features)
        x = self.output_head(x)
        return x


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



def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler, verbose=0):
    """
    Train the model using the provided data loader.

    Args:
        model: The neural network model
        train_loader: DataLoader containing training data
        val_loader: ValLoader
        criterion: Loss function
        optimizer: Optimizer for updating model parameters
        num_epochs: Number of training epochs
        device: Device to train on (cuda/cpu)


    Returns:
        model: Trained model
        history: Dictionary containing training metrics
    """
    model.train()
    history = {
        'loss': [],
        'accuracy': []
    }
    val_history = {
        'val_loss': [],
        'val_delta': [],
    }

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0


        for batch_idx, (num_feats, cat_feats, labels) in enumerate(train_loader):
            # Move data to device
            num_feats = [feat.to(device) for feat in num_feats]
            cat_feats = [feat.to(device) for feat in cat_feats]
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(num_feats, cat_feats)

            # Reshape labels if necessary (e.g., if they're not already in the right shape)
            labels = labels.squeeze()

            # Calculate loss
            loss = criterion(outputs, labels.long())

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Print progress
            # if batch_idx % 10 == 0:
            #     print(f'Epoch: {epoch}, Batch: {batch_idx}, '
            #           f'Loss: {loss.item():.4f}, '
            #           f'Acc: {100. * correct / total:.2f}%')

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        # Store metrics
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)


        if verbose == 1:
            print(f'Epoch {epoch} finished. '
                f'Loss: {epoch_loss:.4f}, '
                f'Accuracy: {epoch_acc:.2f}%')

        val_loss, val_acc, _ = evaluate_model(model, val_loader, criterion, device)
        
        # if epoch > 20: 
        #     scheduler.step(val_loss)

        val_history['val_loss'].append(val_loss)
        scheduler.step()
        # if epoch > 2:
        #     val_history['val_delta'].append(val_history['val_loss'][-1] - val_history['val_loss'][-2])

        # if epoch > 40:
        #     if np.mean(val_history['val_delta'][-10:]) < 0.0005:
        #         print("Early stopping")
        #         break

        if epoch % 5 == 0:
            print(f'Learning Rate: {scheduler.optimizer.param_groups[0]["lr"]:.8f}',
                  f'Val Loss: {val_loss:.4f}, '
                  f'Val Accuracy: {val_acc:.2f}%')


    return model, history



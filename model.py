import torch
from mambular.base_models import BaseModel
from torch.utils.data import Dataset, DataLoader
from torch import nn

class Model(BaseModel):
    def __init__(
            self,
            cat_feature_info,
            num_feature_info,
            model,
            output_dim=32,
            pretrained_state_dict=None,
            config = None,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])
        if config is not None:
            self.model = model(cat_feature_info, num_feature_info, output_dim, config=config)
        else:
            self.model = model(cat_feature_info, num_feature_info, output_dim)

        self.output_dim = output_dim
        self.state_dict = pretrained_state_dict
        if self.state_dict is not None:
            self.model.load_state_dict(self.state_dict)  # Fixed loading state dict
            # Update output_dim based on the model's actual output dimension

        self.output_head =     nn.Sequential(  
                                            #    nn.SELU(),
                                            #    nn.Linear(32, 16),
                                            #    nn.SELU(),
                                            #    nn.Linear(16, 2),
                                               nn.Sigmoid())


        # self.output_activation = nn.Sigmoid()

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
            _, predicted = outputs.max(1)
            total_samples += labels.size(0)
            total_correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
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
        'val_accuracy': [],
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
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100. * correct / total:.2f}%')

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        # Store metrics
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)

        print(f'Epoch {epoch} finished. '
              f'Loss: {epoch_loss:.4f}, '
              f'Accuracy: {epoch_acc:.2f}%')

        if epoch % 10 == 0:
            avg_loss, accuracy = evaluate_model(model, val_loader, criterion, device)
            val_history['val_loss'].append(avg_loss)
            val_history['val_accuracy'].append(accuracy)
            print( f'Val Loss: {avg_loss:.4f}, '
                  f'Val Accuracy: {accuracy:.2f}%')

    return model, history



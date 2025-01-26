import torch
from mambular.base_models import BaseModel
from torch.utils.data import Dataset, DataLoader
from torch import nn
from mambular.configs import DefaultFTTransformerConfig

class Model(BaseModel):
    def __init__(
            self,
            cat_feature_info,
            num_feature_info,
            model,
            output_dim,
            pretrained_state_dict=None,
            config = None,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])
        if config is None:
            config = DefaultFTTransformerConfig()
        self.model = model(cat_feature_info, num_feature_info, output_dim, config)
        self.output_dim = output_dim

        self.state_dict = pretrained_state_dict
        if self.state_dict is not None:
            self.model.load_state_dict(self.state_dict)  # Fixed loading state dict

        if self.output_dim > 2:
            self.output_head =     nn.Sequential(nn.SELU(),
                                                 nn.Linear(output_dim, 16),
                                                 nn.SELU(),
                                                 nn.Linear(16, 2), 
                                                 nn.Sigmoid())
        else:
            self.output_head = nn.Sigmoid()

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

# def evaluate_model(model, data_loader, criterion, device):
#     """Evaluate the model on given data loader."""
#     model.eval()
#     total_loss = 0
#     total_samples = 0
#     total_correct = 0
#     preds = []
#     with torch.no_grad():
#         for num_features, cat_features, labels in data_loader:
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


# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler):
#     """
#     Train the model using the provided data loader.

#     Args:
#         model: The neural network model
#         train_loader: DataLoader containing training data
#         val_loader: ValLoader
#         criterion: Loss function
#         optimizer: Optimizer for updating model parameters
#         num_epochs: Number of training epochs
#         device: Device to train on (cuda/cpu)


#     Returns:
#         model: Trained model
#         history: Dictionary containing training metrics
#     """
#     model.train()
#     history = {
#         'loss': [],
#         'accuracy': []
#     }
#     val_history = {
#         'val_loss': [],
#         'val_accuracy': [],
#     }

#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         correct = 0
#         total = 0


#         for batch_idx, (num_feats, cat_feats, labels) in enumerate(train_loader):
#             # Move data to device
#             num_feats = [feat.to(device) for feat in num_feats]
#             cat_feats = [feat.to(device) for feat in cat_feats]
#             labels = labels.to(device)

#             # Zero the parameter gradients
#             optimizer.zero_grad()

#             # Forward pass
#             outputs = model(num_feats, cat_feats)

#             # Reshape labels if necessary (e.g., if they're not already in the right shape)
#             labels = labels.squeeze()

#             # Calculate loss
#             loss = criterion(outputs, labels.long())

#             # Backward pass and optimize
#             loss.backward()
#             optimizer.step()

#             # Statistics
#             running_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()

#             # Print progress
#             # if batch_idx % 10 == 0:
#             #     print(f'Epoch: {epoch}, Batch: {batch_idx}, '
#             #           f'Loss: {loss.item():.4f}, '
#             #           f'Acc: {100. * correct / total:.2f}%')

#         # Calculate epoch metrics
#         scheduler.step()
#         epoch_loss = running_loss / len(train_loader)
#         epoch_acc = 100. * correct / total

#         # Store metrics
#         history['loss'].append(epoch_loss)
#         history['accuracy'].append(epoch_acc)

#         print(f'Epoch {epoch} finished. '
#               f'Loss: {epoch_loss:.4f}, '
#               f'Accuracy: {epoch_acc:.2f}%')

#         if epoch % 5 == 0:
#             val_loss, val_acc, _ = evaluate_model(model, val_loader, criterion, device)
#             print(f'Val Loss: {val_loss:.4f}, '
#                   f'Val Accuracy: {val_acc:.2f}%')


#     return model, history


def evaluate_models(models, dataloaders, criterions, device):
    streams = [torch.cuda.Stream() for _ in range(len(models))]
    results = []
    
    for model, loader, criterion, stream in zip(models, dataloaders, criterions, streams):
        model.eval()
        with torch.no_grad(), torch.cuda.stream(stream):
            total_loss = 0
            total_correct = 0
            total_samples = 0
            preds = []
            
            for num_features, cat_features, labels in loader:
                num_features = [f.to(device) for f in num_features]
                cat_features = [f.to(device) for f in cat_features]
                labels = labels.to(device).squeeze()
                
                outputs = model(num_features, cat_features)
                loss = criterion(outputs, labels.long())
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total_samples += labels.size(0)
                total_correct += predicted.eq(labels).sum().item()
                preds.append(outputs[:, 1])
                
            epoch_loss = total_loss / len(loader)
            epoch_acc = 100. * total_correct / total_samples
            preds = torch.cat(preds)
            results.append((epoch_loss, epoch_acc, preds))
    
    torch.cuda.synchronize()
    return results

def train_models(models, train_loaders, val_loaders, criterions, optimizers, num_epochs, device, schedulers):
    streams = [torch.cuda.Stream() for _ in range(len(models))]
    histories = [{'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []} for _ in models]
    
    for epoch in range(num_epochs):
        # Training
        for model, loader, criterion, optimizer, stream in zip(models, train_loaders, criterions, optimizers, streams):
            model.train()
            with torch.cuda.stream(stream):
                running_loss = 0.0
                correct = 0
                total = 0
                
                for num_feats, cat_feats, labels in loader:
                    num_feats = [f.to(device) for f in num_feats]
                    cat_feats = [f.to(device) for f in cat_feats]
                    labels = labels.to(device).squeeze()
                    
                    optimizer.zero_grad()
                    outputs = model(num_feats, cat_feats)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
        
        torch.cuda.synchronize()
        
        # Update schedulers and histories
        for i, (model, val_loader, criterion, scheduler, history) in enumerate(zip(models, val_loaders, criterions, schedulers, histories)):
            scheduler.step()
            epoch_loss = running_loss / len(train_loaders[i])
            epoch_acc = 100. * correct / total
            
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            
            if epoch % 5 == 0:
                val_loss, val_acc, _ = evaluate_models([model], [val_loader], [criterion], device)[0]
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                print(f'Model {i}: Epoch {epoch}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return models, histories
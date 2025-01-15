import torch
import torch.nn.functional as F
from mambular.base_models import BaseModel
from torch import nn


class ECELoss(torch.nn.Module):
    def __init__(self, n_bins=10):
        super(ECELoss, self).__init__()
        self.n_bins = n_bins
        self.bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    def forward(self, logits, labels):
        """
        Differentiable ECE Loss for binary classification
        Args:
            logits: Tensor of shape [batch_size, 1]
            labels: Tensor of shape [batch_size, 1]
        """
        # Get predicted probabilities
        probs = torch.sigmoid(logits)

        # Initialize ECE
        ece = torch.zeros(1, device=logits.device, requires_grad=True)

        # Convert to binary accuracies (0 or 1)
        predictions = (probs > 0.5).float()
        accuracies = (predictions == labels).float()

        # Compute ECE for each bin
        for bin_lower, bin_upper in zip(self.bin_boundaries[:-1], self.bin_boundaries[1:]):
            # Find samples in bin
            in_bin = torch.logical_and(probs > bin_lower, probs <= bin_upper)

            if in_bin.any():
                # Calculate average confidence and accuracy in bin
                prop_in_bin = in_bin.float().mean()
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = probs[in_bin].mean()

                # Add weighted absolute difference to ECE
                ece = ece + prop_in_bin * torch.abs(avg_confidence_in_bin - accuracy_in_bin)

        return ece

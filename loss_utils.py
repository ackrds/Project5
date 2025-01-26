import torch
import torch.nn as nn
import torch.nn.functional as F


class SCELoss(nn.Module):
    """
    Static Calibration Error Loss.
    Calculates calibration error separately for each class probability and averages across bins.
    This is an extension of ECE that handles the full multiclass probability distribution.
    """
    def __init__(self, n_bins=15, n_classes=2):
        """
        Args:
            n_bins (int): Number of confidence interval bins
            n_classes (int): Number of classes in the prediction task
        """
        super(SCELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.n_bins = n_bins
        self.n_classes = n_classes

    def forward(self, logits, labels):
        """
        Calculate the Static Calibration Error.

        Args:
            logits: Model output logits of shape (batch_size, n_classes)
            labels: Ground truth labels of shape (batch_size,)

        Returns:
            Scalar tensor containing the SCE loss
        """
        softmaxes = F.softmax(logits, dim=1)  # Convert logits to probabilities

        # Convert labels to one-hot encoding
        labels_one_hot = F.one_hot(labels, num_classes=self.n_classes).float()

        sce = torch.zeros(1, device=logits.device)

        # Calculate calibration error for each class
        for class_idx in range(self.n_classes):
            class_confidences = softmaxes[:, class_idx]  # Get probabilities for current class
            class_accuracies = labels_one_hot[:, class_idx]  # Get true labels for current class

            # Calculate calibration error across bins for this class
            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                # Calculate the confidence masks and filter out samples
                in_bin = (class_confidences > bin_lower) & (class_confidences <= bin_upper)
                prop_in_bin = in_bin.float().mean()

                if prop_in_bin.item() > 0:
                    # Calculate accuracy and confidence in this bin
                    accuracy_in_bin = class_accuracies[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()

                    # Add weighted absolute difference to SCE
                    sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        # Average across classes
        sce = sce / self.n_classes

        return sce


class HybridLoss(nn.Module):
    """
    Combines cross entropy loss with SCE loss for better calibration during training
    """

    def __init__(self, n_bins=15, n_classes=2, ce_weight = 0.1, sce_weight=0.1):
        """
        Args:
            n_bins (int): Number of bins for SCE calculation
            n_classes (int): Number of classes in the prediction task
            sce_weight (float): Weight for SCE loss in the combined loss function
        """
        super(HybridLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.sce_criterion = SCELoss(n_bins=n_bins, n_classes=n_classes)
        self.ce_weight = ce_weight
        self.sce_weight = sce_weight

    def forward(self, logits, labels):
        """
        Calculate the combined loss.

        Args:
            logits: Model output logits of shape (batch_size, n_classes)
            labels: Ground truth labels of shape (batch_size,)

        Returns:
            Combined loss of CrossEntropy and SCE
        """
        ce_loss = self.cross_entropy(logits, labels)
        sce_loss = self.sce_criterion(logits, labels)

        return self.ce_weight * ce_loss  + self.sce_weight*sce_loss

"""Loss functions for handling class imbalance."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    https://arxiv.org/abs/1708.02002

    Focal loss down-weights easy examples and focuses on hard examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """Initialize focal loss.

        Args:
            alpha: Weighting factor in [0, 1] for class 1 (unfair)
                  Higher alpha gives more weight to positive class
            gamma: Focusing parameter >= 0. Higher gamma down-weights easy examples more
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Model predictions of shape (batch_size, 1)
            targets: Ground truth labels of shape (batch_size,)

        Returns:
            Focal loss value
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(logits.squeeze(1))

        # Get targets as float
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy(probs, targets, reduction='none')

        # Compute p_t (probability of correct class)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Compute alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Final focal loss
        focal_loss = alpha_t * focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """Binary cross-entropy loss with class weights."""

    def __init__(self, pos_weight: float = 1.0):
        """Initialize weighted BCE loss.

        Args:
            pos_weight: Weight for positive class (unfair)
                       Higher value gives more weight to unfair examples
        """
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted BCE loss.

        Args:
            logits: Model predictions of shape (batch_size, 1)
            targets: Ground truth labels of shape (batch_size,)

        Returns:
            Weighted BCE loss value
        """
        targets = targets.float().unsqueeze(1)
        self.pos_weight = self.pos_weight.to(logits.device)

        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight
        )

        return loss


def get_loss_function(
    use_focal_loss: bool = True,
    use_class_weights: bool = True,
    class_weights: torch.Tensor = None,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0
) -> nn.Module:
    """Get appropriate loss function based on configuration.

    Args:
        use_focal_loss: Whether to use focal loss
        use_class_weights: Whether to use class weights
        class_weights: Tensor of class weights [weight_fair, weight_unfair]
        focal_alpha: Focal loss alpha parameter
        focal_gamma: Focal loss gamma parameter

    Returns:
        Loss function module
    """
    if use_focal_loss:
        print(f"Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    elif use_class_weights and class_weights is not None:
        # pos_weight is ratio of negative to positive class weights
        pos_weight = class_weights[1] / class_weights[0]
        print(f"Using Weighted BCE Loss (pos_weight={pos_weight:.3f})")
        return WeightedBCELoss(pos_weight=pos_weight.item())

    else:
        print("Using standard BCE Loss")
        return nn.BCEWithLogitsLoss()

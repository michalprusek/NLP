"""LaMCTS: Latent Action Monte Carlo Tree Search.

Implements:
- LaMCTSOptimizer: Tree-based partitioning without GP

LaMCTS partitions the search space using a tree structure, avoiding
the O(N³) complexity of GP. Each leaf contains samples and maintains
simple statistics for UCB-based selection.

References:
- Wang et al. (2020) "Learning Search Space Partition for Black-box Optimization"
- Wang et al. (2020) "Learning to Partition for Black-Box Optimization" (NeurIPS)
"""

import logging
import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field

from study.gp_ablation.config import GPConfig
from study.gp_ablation.surrogates.base import BaseGPSurrogate

logger = logging.getLogger(__name__)


@dataclass
class MCTSNode:
    """Node in the MCTS tree.

    Each node represents a region of the search space, defined by
    a splitting hyperplane.
    """

    # Node identification
    depth: int = 0
    node_id: int = 0

    # Splitting hyperplane: points with (x - center) · direction > 0 go right
    split_direction: Optional[torch.Tensor] = None
    split_center: Optional[torch.Tensor] = None

    # Children
    left: Optional["MCTSNode"] = None
    right: Optional["MCTSNode"] = None

    # Samples in this region (leaf nodes only)
    X: List[torch.Tensor] = field(default_factory=list)
    Y: List[float] = field(default_factory=list)

    # UCB statistics
    n_visits: int = 0
    value_sum: float = 0.0
    max_value: float = float("-inf")

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    @property
    def mean_value(self) -> float:
        if self.n_visits == 0:
            return 0.0
        return self.value_sum / self.n_visits

    @property
    def ucb(self) -> float:
        """Upper Confidence Bound for node selection."""
        if self.n_visits == 0:
            return float("inf")  # Explore unvisited nodes first

        # UCB1 formula
        exploration = math.sqrt(2 * math.log(self._parent_visits + 1) / self.n_visits)
        return self.mean_value + 2.0 * exploration

    _parent_visits: int = 1  # Set during tree traversal


class LaMCTSOptimizer(BaseGPSurrogate):
    """LaMCTS optimizer without GP.

    Uses Monte Carlo Tree Search to partition the space and guide
    exploration. Does not use GP, avoiding O(N³) scaling.

    Key hyperparameters:
        max_depth: Maximum tree depth
        n_samples_per_leaf: Minimum samples before splitting a leaf
    """

    def __init__(self, config: GPConfig, device: torch.device | str = "cuda"):
        super().__init__(config, device)

        self.max_depth = config.max_depth
        self.n_samples_per_leaf = config.n_samples_per_leaf
        self.model = None  # LaMCTS doesn't use GP model

        # Tree root
        self.root: Optional[MCTSNode] = None
        self._node_counter = 0

    def _create_node(self, depth: int = 0) -> MCTSNode:
        """Create a new MCTS node."""
        node = MCTSNode(depth=depth, node_id=self._node_counter)
        self._node_counter += 1
        return node

    def _select_split_direction(self, X: List[torch.Tensor]) -> torch.Tensor:
        """Select splitting direction using PCA on local samples."""
        if len(X) < 2:
            return F.normalize(torch.randn(self.D, device=self.device), dim=0)

        X_tensor = torch.stack(X)
        X_centered = X_tensor - X_tensor.mean(dim=0, keepdim=True)

        try:
            U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
            return Vh[0]  # First principal component
        except RuntimeError:
            return F.normalize(torch.randn(self.D, device=self.device), dim=0)

    def _split_node(self, node: MCTSNode) -> None:
        """Split a leaf node into two children."""
        if not node.is_leaf or len(node.X) < 2:
            return

        # Compute splitting hyperplane
        X_tensor = torch.stack(node.X)
        direction = self._select_split_direction(node.X)
        center = X_tensor.mean(dim=0)

        node.split_direction = direction
        node.split_center = center

        # Create children
        node.left = self._create_node(depth=node.depth + 1)
        node.right = self._create_node(depth=node.depth + 1)

        # Distribute samples
        for x, y in zip(node.X, node.Y):
            if self._goes_left(x, node):
                node.left.X.append(x)
                node.left.Y.append(y)
                node.left.n_visits += 1
                node.left.value_sum += y
                node.left.max_value = max(node.left.max_value, y)
            else:
                node.right.X.append(x)
                node.right.Y.append(y)
                node.right.n_visits += 1
                node.right.value_sum += y
                node.right.max_value = max(node.right.max_value, y)

        # Clear samples from internal node
        node.X = []
        node.Y = []

    def _goes_left(self, x: torch.Tensor, node: MCTSNode) -> bool:
        """Check if x goes to left child."""
        if node.split_direction is None:
            return True
        proj = (x - node.split_center) @ node.split_direction
        return proj.item() <= 0

    def _select_leaf(self, root: MCTSNode) -> MCTSNode:
        """Select a leaf node using UCB."""
        node = root

        while not node.is_leaf:
            # Set parent visits for UCB calculation
            node.left._parent_visits = node.n_visits
            node.right._parent_visits = node.n_visits

            # Select child with higher UCB
            if node.left.ucb >= node.right.ucb:
                node = node.left
            else:
                node = node.right

        return node

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagate value up the tree."""
        # Note: This simple implementation doesn't maintain parent pointers
        # In practice, we update during forward traversal
        pass

    def fit(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Initialize tree with training data."""
        self._train_X = train_X.to(self.device)
        self._train_Y = train_Y.to(self.device)

        if self._train_Y.dim() > 1:
            self._train_Y = self._train_Y.squeeze(-1)

        # Create root node with all data
        self.root = self._create_node(depth=0)

        for x, y in zip(self._train_X, self._train_Y):
            self.root.X.append(x)
            self.root.Y.append(y.item())
            self.root.n_visits += 1
            self.root.value_sum += y.item()
            self.root.max_value = max(self.root.max_value, y.item())

        # Build initial tree by splitting
        self._build_tree(self.root)

    def _build_tree(self, node: MCTSNode) -> None:
        """Recursively build tree by splitting nodes."""
        if node.depth >= self.max_depth:
            return

        if len(node.X) >= self.n_samples_per_leaf * 2:
            self._split_node(node)

            if node.left:
                self._build_tree(node.left)
            if node.right:
                self._build_tree(node.right)

    def update(self, new_X: torch.Tensor, new_Y: torch.Tensor) -> None:
        """Add new samples to the tree."""
        if self.root is None:
            self.fit(new_X, new_Y)
            return

        new_X = new_X.to(self.device)
        new_Y = new_Y.to(self.device)
        if new_Y.dim() > 1:
            new_Y = new_Y.squeeze(-1)

        # Add to training data
        self._train_X = torch.cat([self._train_X, new_X], dim=0)
        self._train_Y = torch.cat([self._train_Y, new_Y], dim=0)

        # Insert each new sample into tree
        for x, y in zip(new_X, new_Y):
            self._insert(self.root, x, y.item())

    def _insert(self, node: MCTSNode, x: torch.Tensor, y: float) -> None:
        """Insert a sample into the tree."""
        node.n_visits += 1
        node.value_sum += y
        node.max_value = max(node.max_value, y)

        if node.is_leaf:
            node.X.append(x)
            node.Y.append(y)

            # Check if we should split
            if (
                node.depth < self.max_depth
                and len(node.X) >= self.n_samples_per_leaf * 2
            ):
                self._split_node(node)
        else:
            # Recurse to appropriate child
            if self._goes_left(x, node):
                self._insert(node.left, x, y)
            else:
                self._insert(node.right, x, y)

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict using leaf statistics.

        For each point, finds its leaf and returns (mean, std) of samples in that leaf.
        """
        if self.root is None:
            raise RuntimeError("LaMCTS not fitted. Call fit() first.")

        X = X.to(self.device)
        means = []
        stds = []

        for x in X:
            leaf = self._find_leaf(self.root, x)

            if len(leaf.Y) > 0:
                Y_tensor = torch.tensor(leaf.Y, device=self.device)
                means.append(Y_tensor.mean())
                stds.append(Y_tensor.std() if len(leaf.Y) > 1 else torch.tensor(1.0))
            else:
                # Empty leaf - use global statistics
                means.append(self._train_Y.mean())
                stds.append(self._train_Y.std())

        return torch.stack(means), torch.stack(stds)

    def _find_leaf(self, node: MCTSNode, x: torch.Tensor) -> MCTSNode:
        """Find the leaf node containing x."""
        while not node.is_leaf:
            if self._goes_left(x, node):
                node = node.left
            else:
                node = node.right
        return node

    def suggest(
        self,
        n_candidates: int = 1,
        bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        n_samples: int = 512,
    ) -> torch.Tensor:
        """Suggest candidates using MCTS selection + local sampling.

        1. Select promising leaf via UCB
        2. Sample from that region
        """
        if self.root is None:
            # Not fitted - return random samples
            return torch.randn(n_candidates, self.D, device=self.device)

        suggestions = []

        for _ in range(n_candidates):
            # Select leaf using UCB
            leaf = self._select_leaf(self.root)

            # Sample from leaf region
            if len(leaf.X) > 0:
                # Sample near the best point in leaf
                X_leaf = torch.stack(leaf.X)
                Y_leaf = torch.tensor(leaf.Y, device=self.device)
                best_idx = Y_leaf.argmax()

                # Gaussian perturbation around best
                mean = X_leaf[best_idx]
                std = X_leaf.std(dim=0).clamp(min=0.01)
                sample = mean + std * torch.randn(self.D, device=self.device)
            else:
                # Empty leaf - sample from global distribution
                sample = self._train_X.mean(dim=0) + self._train_X.std(dim=0) * torch.randn(
                    self.D, device=self.device
                )

            suggestions.append(sample)

        return torch.stack(suggestions)

    def get_tree_stats(self) -> dict:
        """Get statistics about the tree structure."""
        if self.root is None:
            return {"n_nodes": 0, "n_leaves": 0, "max_depth": 0}

        def traverse(node, depth=0):
            if node.is_leaf:
                return 1, 1, depth

            left_nodes, left_leaves, left_depth = traverse(node.left, depth + 1)
            right_nodes, right_leaves, right_depth = traverse(node.right, depth + 1)

            return (
                1 + left_nodes + right_nodes,
                left_leaves + right_leaves,
                max(left_depth, right_depth),
            )

        n_nodes, n_leaves, max_depth = traverse(self.root)

        return {
            "n_nodes": n_nodes,
            "n_leaves": n_leaves,
            "max_depth": max_depth,
            "avg_samples_per_leaf": self.n_train / n_leaves if n_leaves > 0 else 0,
        }

    def _ensure_fitted(self, operation: str = "operation") -> None:
        """Check if tree is built instead of GP model.

        Overrides base class since LaMCTS uses tree instead of GP.
        """
        if self.root is None:
            raise RuntimeError(f"LaMCTS must be fitted before {operation}. Call fit() first.")

    def compute_acquisition(
        self,
        X: torch.Tensor,
        acquisition: str = "log_ei",
        best_f: Optional[float] = None,
        alpha: float = 1.96,
    ) -> torch.Tensor:
        """Compute acquisition using tree-based UCB.

        LaMCTS uses tree statistics instead of GP posterior.
        For each point, finds its leaf and returns UCB based on leaf statistics.

        Args:
            X: Candidate points [M, D].
            acquisition: Ignored (always uses UCB).
            best_f: Ignored.
            alpha: UCB exploration weight.

        Returns:
            Acquisition values [M].
        """
        self._ensure_fitted("computing acquisition")

        X = X.to(self.device)
        acq_values = []

        for x in X:
            leaf = self._find_leaf(self.root, x)

            if len(leaf.Y) > 0:
                Y_tensor = torch.tensor(leaf.Y, device=self.device)
                mean = Y_tensor.mean()
                std = Y_tensor.std() if len(leaf.Y) > 1 else torch.tensor(0.1, device=self.device)
            else:
                # Empty leaf - use global statistics
                mean = self._train_Y.mean()
                std = self._train_Y.std()

            # UCB formula
            ucb = mean + alpha * std
            acq_values.append(ucb)

        return torch.stack(acq_values)

    def acquisition_gradient(
        self,
        X: torch.Tensor,
        acquisition: str = "ucb",
        alpha: float = 1.96,
    ) -> torch.Tensor:
        """Compute gradient of acquisition function.

        LaMCTS doesn't support gradients - returns zeros.

        Args:
            X: Input points [B, D].
            acquisition: Ignored.
            alpha: Ignored.

        Returns:
            Zero gradient tensor [B, D].
        """
        return torch.zeros_like(X, device=self.device)

    def sample_posterior(
        self, X: torch.Tensor, n_samples: int = 1
    ) -> torch.Tensor:
        """Draw samples from leaf distribution.

        LaMCTS doesn't have a full posterior - samples from leaf statistics.

        Args:
            X: Test inputs [M, D].
            n_samples: Number of samples.

        Returns:
            Samples [n_samples, M].
        """
        self._ensure_fitted("sampling")

        X = X.to(self.device)
        samples = []

        for _ in range(n_samples):
            sample_row = []
            for x in X:
                leaf = self._find_leaf(self.root, x)

                if len(leaf.Y) > 0:
                    Y_tensor = torch.tensor(leaf.Y, device=self.device)
                    mean = Y_tensor.mean()
                    std = Y_tensor.std() if len(leaf.Y) > 1 else torch.tensor(0.1, device=self.device)
                else:
                    mean = self._train_Y.mean()
                    std = self._train_Y.std()

                # Sample from Gaussian approximation
                sample = mean + std * torch.randn(1, device=self.device)
                sample_row.append(sample)

            samples.append(torch.cat(sample_row))

        return torch.stack(samples)

    def get_hyperparameters(self) -> dict:
        """Get tree hyperparameters instead of GP hyperparameters.

        Returns:
            Dict with tree statistics.
        """
        stats = self.get_tree_stats()
        return {
            "lengthscale_mean": None,  # No lengthscale in tree
            "lengthscale_std": None,
            "n_nodes": stats.get("n_nodes", 0),
            "n_leaves": stats.get("n_leaves", 0),
            "max_depth": stats.get("max_depth", 0),
        }

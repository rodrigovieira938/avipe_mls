from . import constants

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseLoss(nn.Module):
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ComposeLoss(BaseLoss):
    def __init__(self, children: list[BaseLoss], weight:float):
        super().__init__()
        self._children = children
        self.weight = weight
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((), device=outputs.device)

        for loss in self._children:
            total = total + loss(outputs, targets)

        return self.weight * total
class CrossEntropyLoss(BaseLoss):
    def __init__(
        self,
        class_weights: list[float] | None = None,
        weight: float = 1.0,
        ignore_index: int | None = None,
    ):
        super().__init__()

        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(constants.TORCH_DEVICE)
        else:
            weight_tensor = None

        self.criterion = nn.CrossEntropyLoss(
            weight=weight_tensor,
            ignore_index=ignore_index if ignore_index is not None else -100,
        )
        self.weight = weight

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        outputs: [B, C, H, W] (logits)
        targets: [B, 1, H, W] or [B, H, W]
        """

        if targets.ndim == 4:
            targets = targets.squeeze(1)

        targets = targets.long()

        loss = self.criterion(outputs, targets)
        return self.weight * loss
class DiceLoss(BaseLoss):
    def __init__(
        self,
        weight: float = 1.0,
        class_weights: list[float] | None = None,
        ignore_index: int | None = None,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.smooth = smooth

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        outputs: [B, C, H, W] (logits)
        targets: [B, H, W] or [B, 1, H, W]
        """

        if targets.ndim == 4:
            targets = targets.squeeze(1)

        targets = targets.long()
        num_classes = outputs.shape[1]

        # Softmax over classes
        probs = F.softmax(outputs, dim=1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(
            torch.clamp(targets, min=0),
            num_classes=num_classes,
        ).permute(0, 3, 1, 2).float()

        # Handle ignore_index
        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index).unsqueeze(1)
            probs = probs * valid_mask
            targets_one_hot = targets_one_hot * valid_mask

        dims = (0, 2, 3)

        intersection = torch.sum(probs * targets_one_hot, dims)
        union = torch.sum(probs + targets_one_hot, dims)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice  # [C]

        if self.class_weights is not None:
            dice_loss = dice_loss * self.class_weights.to(dice_loss.device)

        return self.weight * dice_loss.mean()
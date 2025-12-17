from . import constants

import torch
import torch.nn as nn

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
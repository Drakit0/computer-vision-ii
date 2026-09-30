import torch
from torch import nn

from src.constants import EPSILON, LAMBDA_COORD, LAMBDA_NOOBJ
from src.utils import iou


class YOLOLoss(nn.Module):
    """
    Implementation of YOLO loss function from the original paper. Some parts are
    hardcoded because we assume B = 2.
    """

    def __init__(self) -> None:
        """
        Constructor of the class.
        """

        super().__init__()

        self.mse = nn.MSELoss()

    def _coordinates_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        one_i_obj: torch.Tensor,
        best_box: torch.Tensor,
    ) -> torch.Tensor:
        """
        Obtains the loss for the predicted coordinates of the bounding boxes.

        Args:
            predictions: Predictions of the model. Dimensions: [batch, S, S, 5 * B + C].
            targets: Ground truths. Dimensions: [batch, S, S, 5 * B + C].
            one_i_obj: Binary tensor indicating if there is an object in each cell.
                Dimensions: [batch, S, S, 1].
            best_box: Binary tensor indicating the best box for each cell, which is the
                one with more confidence. For position (i, j): 0 -> first bounding box,
                1 -> second bounding box, because we assume B = 2). Dimensions:
                [batch_size, S, S, 1].

        Returns:
            Loss for the predicted coordinates. It is a scalar.
        """

        predictions_masked: torch.Tensor = predictions * one_i_obj
        targets_masked: torch.Tensor = targets * one_i_obj

        # IDXs coordinates
        start_idx: torch.Tensor = torch.where(
            best_box.bool(),
            torch.tensor(7, device=predictions.device),
            torch.tensor(2, device=predictions.device),
        )  # [batch, S, S, 1]

        offsets: torch.Tensor = torch.arange(4, device=predictions.device)
        predict_idxs: torch.Tensor = start_idx + offsets  # [batch, S, S, 4]
        target_idx: torch.Tensor = (
            torch.zeros(start_idx.shape, device=predictions.device) + 2 + offsets
        )  # [batch, S, S, 4]

        predictions_coords: torch.Tensor = torch.gather(
            predictions_masked, -1, predict_idxs
        )
        targets_coords: torch.Tensor = torch.gather(
            targets_masked, -1, target_idx.int()
        )  # The second box has nothing

        # Separate H, W
        predictions_size: torch.Tensor = predictions_coords[..., 2:]
        targets_size: torch.Tensor = targets_coords[..., 2:]

        predictions_size_signs: torch.Tensor = (
            predictions_size.sign()
        )  # Possible negative predicts

        predictions_size_sqrt: torch.Tensor = predictions_size_signs * torch.sqrt(
            predictions_size.abs() + EPSILON
        )
        targets_size_sqrt: torch.Tensor = torch.sqrt(targets_size + EPSILON)

        # Join the coords and the sqrts
        predictions_clean: torch.Tensor = torch.zeros(predictions_coords.shape)
        predictions_clean[..., :2] = predictions_coords[..., :2]
        predictions_clean[..., 2:] = predictions_size_sqrt

        targets_clean: torch.Tensor = torch.zeros(targets_coords.shape)
        targets_clean[..., :2] = targets_coords[..., :2]
        targets_clean[..., 2:] = targets_size_sqrt

        loss: torch.Tensor = self.mse(predictions_clean, targets_clean)

        return LAMBDA_COORD * loss

    def _object_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        one_i_obj: torch.Tensor,
        best_box: torch.Tensor,
    ) -> torch.Tensor:
        """
        Obtains the loss for the prediction of if a cell contains an object.

        Args:
            predictions: Predictions of the model. Dimensions: [batch, S, S, 5 * B + C].
            targets: Ground truths. Dimensions: [batch, S, S, 5 * B + C].
            one_i_obj: Binary tensor indicating if there is an object in each cell.
                Dimensions: [batch, S, S, 1].
            best_box: Binary tensor indicating the best box for each cell, which is the
                one with more confidence. For position (i, j): 0 -> first bounding box,
                1 -> second bounding box, because we assume B = 2). Dimensions:
                [batch_size, S, S, 1].

        Returns:
            Loss for the prediction of if a cell contains an object. It is a scalar.
        """

        predictions_masked: torch.Tensor = predictions * one_i_obj
        targets_masked: torch.Tensor = targets * one_i_obj

        # Extract correct idx
        idx: torch.Tensor = torch.where(
            best_box.bool(), torch.tensor(6), torch.tensor(1)
        )

        confidence_predictions: torch.Tensor = torch.gather(
            predictions_masked, -1, idx
        )  # [batch, S, S, 1]
        confidence_targets: torch.Tensor = torch.gather(
            targets_masked,
            -1,
            torch.zeros(idx.shape, dtype=torch.int, device=predictions.device) + 1,
        )  # [batch, S, S, 1]

        loss: torch.Tensor = self.mse(confidence_predictions, confidence_targets)

        return loss

    def _no_object_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        one_i_obj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Obtains the loss for the prediction of if a cell does not contain an object.

        Args:
            predictions: Predictions of the model. Dimensions: [batch, S, S, 5 * B + C].
            targets: Ground truths. Dimensions: [batch, S, S, 5 * B + C].
            one_i_obj: Binary tensor indicating if there is an object in each cell.
                Dimensions: [batch, S, S].

        Returns:
            Loss for the prediction of if a cell does not contain an object. It is a
            scalar.
        """

        no_i_obj: torch.Tensor = (
            torch.ones(one_i_obj.shape, device=predictions.device) - one_i_obj
        )

        predictions_masked: torch.Tensor = predictions * no_i_obj
        targets_masked: torch.Tensor = targets * no_i_obj

        # Loss for bounding boxes when no object around
        loss_box1: torch.Tensor = self.mse(
            predictions_masked[..., 1], targets_masked[..., 1]
        )
        loss_box2: torch.Tensor = self.mse(
            predictions_masked[..., 6], targets_masked[..., 1]
        )

        loss: torch.Tensor = loss_box1 + loss_box2

        return LAMBDA_NOOBJ * loss

    def _class_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        one_i_obj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Obtains the loss for the class of the objects.

        Args:
            predictions: Predictions of the model. Dimensions: [batch, S, S, 5 * B + C].
            targets: Ground truths. Dimensions: [batch, S, S, 5 * B + C].
            one_i_obj: Binary tensor indicating if there is an object in each cell.
                Dimensions: [batch, S, S, 1].

        Returns:
            Loss for the prediction of the class of the objects. It is a scalar.
        """

        predictions_masked: torch.Tensor = predictions * one_i_obj
        targets_masked: torch.Tensor = targets * one_i_obj

        loss: torch.Tensor = self.mse(
            predictions_masked[..., 0], targets_masked[..., 0]
        )

        return loss

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            predictions: Predictions of the model. Dimensions: [batch, S, S, 5 * B + C].
            targets: Ground truths. Dimensions: [batch, S, S, 5 * B + C].

        Returns:
            Each of the sub-losses in the order they are coded and final YOLO loss.
        """

        one_i_obj: torch.Tensor = targets[..., 1].unsqueeze(-1)

        first_predict: torch.Tensor = predictions[..., 2:6]
        second_predict: torch.Tensor = predictions[..., 7:11]
        target_box: torch.Tensor = targets[..., 2:6]

        best_box: torch.Tensor = torch.where(
            iou(first_predict, target_box) < iou(second_predict, target_box),
            torch.tensor(1, device=predictions.device),
            torch.tensor(0, device=predictions.device),
        )

        coord_loss: torch.Tensor = self._coordinates_loss(
            predictions, targets, one_i_obj, best_box
        )
        obj_loss: torch.Tensor = self._object_loss(
            predictions, targets, one_i_obj, best_box
        )
        no_obj_loss: torch.Tensor = self._no_object_loss(
            predictions, targets, one_i_obj
        )
        cls_loss: torch.Tensor = self._class_loss(predictions, targets, one_i_obj)
        yolo_loss: torch.Tensor = coord_loss + obj_loss + no_obj_loss + cls_loss

        return coord_loss, obj_loss, no_obj_loss, cls_loss, yolo_loss

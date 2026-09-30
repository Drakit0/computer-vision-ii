import torch
from src.utils import visualize_perturbations

device = "cuda" if torch.cuda.is_available() else "cpu"


class FastGradientSignMethod:
    """
    This class implements the white-box adversarial attack Fast Gradient Sign Method
    (FGSM): x' = x + epsilon * dL/dx.
    """

    def __init__(self, model: torch.nn.Module, loss: torch.nn.Module) -> None:
        """
        Constructor of the class.

        Parameters
        ----------
        model : Model used.
        loss  : Loss used by the model.
        """

        self.model: torch.nn.Module = model
        self.loss: torch.nn.Module = loss

    def _get_perturbations(self, img: torch.Tensor, label: int) -> torch.Tensor:
        """
        Obtains the gradient of the loss with respect to the input.

        Parameters
        ----------
        img   : Original image. Dimensions: [channels, height, width].
        label : Real label of the image.

        Returns
        -------
        Perturbation for the image. Dimensions: [channels, height, width].
        """

        img.requires_grad_()

        output: torch.Tensor = self.model(
            img.unsqueeze(0).to(device)
        ).squeeze(0) # Add B and remove B

        target: torch.Tensor = torch.tensor(label, device=device)
        loss: torch.Tensor = self.loss(output, target)

        loss.backward()

        return torch.sign(img.grad).to(device)


    def perturb_img(
        self,
        img: torch.Tensor,
        label: int,
        epsilon: float = 1e-2,
        show: bool = True,
        **kwargs_visualize
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perturbs an image.

        Parameters
        ----------
        img     : Original image.
        label   : Real label of the image.
        epsilon : Epsilon parameter.
        show    : Boolean parameter that decides whether to save the figure or not.

        Returns
        -------
        perturbed_img : Perturbed image. Dimensions: [channels, height, width].
        perturbations : Perturbations made. Dimensions: [channels, height, width].
        """

        perturbations: torch.Tensor = epsilon * self._get_perturbations(img, label)
        perturbated_img: torch.Tensor = (img + perturbations).to(device)

        if show:
            visualize_perturbations(
                perturbated_img.detach(),
                img.detach(),
                label,
                self.model,
                **kwargs_visualize
            )

        return perturbated_img, perturbations
# src/model_handler.py
from torch import nn
from torchvision import models
from src.utils import get_final_layer, set_nested_attr


class ModelHandler:
    """Handles model loading and modification for fine-tuning.

    This class provides functionality to load pre-trained models from torchvision
    and dynamically modify them for fine-tuning on custom datasets. It automatically
    finds and replaces the final classification layer to match the target number of
    classes.

    Attributes:
        MODEL_REGISTRY (dict): A dictionary mapping model names to their corresponding
            torchvision model classes. Currently supports resnet18, vgg16, and alexnet.
        config (dict): Configuration dictionary containing model specifications.
        model (torch.nn.Module or None): The loaded PyTorch model instance.

    Example:
        >>> config = {"model": {"name": "resnet18"}}
        >>> handler = ModelHandler(config)
        >>> handler.load_model()
        >>> handler.prepare_for_finetuning(num_classes=10)
    """

    # TODO
    MODEL_REGISTRY = {
        "resnet18": {"model": models.resnet18, "weights": models.ResNet18_Weights},
        "vgg16": {"model": models.vgg16, "weights": models.VGG16_Weights},
        "alexnet": {"model": models.alexnet, "weights": models.AlexNet_Weights},
        "inception": {
            "model": models.inception_v3,
            "weights": models.Inception_V3_Weights,
        },
    }

    def __init__(self, config: dict[str, dict[str, str]]) -> None:
        """Initialize the ModelHandler with configuration settings.

        Args:
            config (dict): Configuration dictionary that must contain a 'model' key
                with a 'name' subkey specifying which model to use. The model name
                must be present in the MODEL_REGISTRY.

        Example:
            >>> config = {
            ...     "model": {
            ...         "name": "resnet18"
            ...     }
            ... }
            >>> handler = ModelHandler(config)
        """

        self.model_name = config["model"]["name"]
        self.model = None

    def load_model(self) -> None:
        """Load a pre-trained model from torchvision based on configuration.

        Loads the specified model with ImageNet pre-trained weights. The model name
        is retrieved from the configuration and must exist in the MODEL_REGISTRY.

        Raises:
            ValueError: If the specified model name is not found in MODEL_REGISTRY.

        Note:
            This method loads models with 'IMAGENET1K_V1' weights, which are the
            standard ImageNet pre-trained weights from torchvision.

        Example:
            >>> handler = ModelHandler({"model": {"name": "resnet18"}})
            >>> handler.load_model()
            Loading pre-trained model: resnet18
        """
        if self.model_name not in self.MODEL_REGISTRY.keys():
            raise ValueError("not found in internal MODEL_REGISTRY")

        self.config = self.MODEL_REGISTRY[self.model_name]

        self.model = self.config["model"](weights=self.config["weights"].IMAGENET1K_V1)

    def replace_last_layer(self, num_classes: int) -> None:
        """Prepare the loaded model for fine-tuning by replacing the final classifier.

        This method dynamically finds the model's final classification layer and
        replaces it with a new linear layer that outputs the specified number of
        classes.
        This is essential for transfer learning when the target dataset has a different
        number of classes than the original pre-trained model
        (e.g., ImageNet's 1000 classes).

        Args:
            num_classes (int): The number of output classes for the new classification
                layer. Must be a positive integer.

        Raises:
            RuntimeError: If no model has been loaded. Must call load_model() first.

        Note:
            The method preserves the input feature dimension of the original final
            layer while only changing the output dimension to match num_classes.
            This ensures compatibility with the pre-trained feature extractor layers.

        Example:
            >>> handler = ModelHandler({"model": {"name": "resnet18"}})
            >>> handler.load_model()
            >>> handler.replace_last_layer(num_classes=5)
            Model 'resnet18' adapted for fine-tuning with 5 classes.
            Replaced layer 'fc' with new classifier:
            Linear(in_features=512, out_features=5, bias=True)
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        last_linear = get_final_layer(self.model)
        set_nested_attr(
            self.model,
            last_linear[0],
            nn.Linear(last_linear[1].in_features, num_classes, bias=True),
        )

# src/strategies.py

from abc import ABC, abstractmethod
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import numpy as np

# Import the helper function from the new utils file
from src.utils import get_final_layer, set_nested_attr


class FineTuningStrategy(ABC):
    """Abstract base class for all fine-tuning strategies.

    This class defines the common structure and shared functionality for different
    fine-tuning strategies. Concrete implementations must define how to prepare
    the model and optimizer for their specific training approach.

    Attributes:
        None - This is an abstract base class with no instance attributes.

    Note:
        Subclasses must implement the _prepare_model_and_optimizer method to define
        their specific fine-tuning approach.

    Example:
        >>> class MyStrategy(FineTuningStrategy):
        ...     def _prepare_model_and_optimizer(self, model, config):
        ...         # Custom preparation logic
        ...         return model, optimizer
        >>> strategy = MyStrategy()
        >>> results = strategy.execute(model, train_loader, val_loader, config)
    """

    def _train_one_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> float:
        """Train the model for one epoch and return the average loss.

        This private helper method performs a single training epoch, iterating through
        the training data, computing losses, performing backpropagation, and updating
        model parameters. Progress is displayed using tqdm.

        Args:
            model (nn.Module): The PyTorch model to train. Should already be on the
                correct device and have appropriate layers unfrozen.
            train_loader (DataLoader): DataLoader providing batches of training data.
                Should yield tuples of (inputs, labels).
            optimizer (torch.optim.Optimizer): Optimizer instance configured with
                the parameters to update.
            device (torch.device): Device (CPU or CUDA) where computations should
                be performed.

        Returns:
            float: Average training loss across all batches in the epoch. Computed
                as the sum of batch losses divided by the number of batches.

        Note:
            The model is set to training mode at the beginning of this method,
            which enables dropout and batch normalization updates.

        Example:
            >>> device = torch.device("cuda")
            >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            >>> avg_loss = self._train_one_epoch(model, train_loader, optimizer, device)
            >>> print(f"Average training loss: {avg_loss:.4f}")
        """

        model.train()
        total_loss = 0.0
        num_batches = 0

        for _, (inputs, labels) in enumerate(tqdm(train_loader, desc="Training")):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _validate(
        self, model: nn.Module, val_loader: DataLoader, device: torch.device
    ) -> dict[str, float]:
        """Evaluate the model on validation data and return performance metrics.

        This private helper method performs model evaluation on the validation set
        without updating parameters. It computes both loss and accuracy metrics.
        Progress is displayed using tqdm.

        Args:
            model (nn.Module): The PyTorch model to evaluate. Should already be on
                the correct device.
            val_loader (DataLoader): DataLoader providing batches of validation data.
                Should yield tuples of (inputs, labels).
            device (torch.device): Device (CPU or CUDA) where computations should
                be performed.

        Returns:
            dict: Dictionary containing validation metrics with the following keys:
                - 'loss' (float): Average validation loss across all batches.
                - 'accuracy' (float): Overall validation accuracy as a fraction
                  between 0 and 1 (correct predictions / total samples).

        Note:
            The model is set to evaluation mode, which disables dropout and uses
            running statistics for batch normalization. Gradients are not computed
            during validation to save memory and computation.

        Example:
            >>> device = torch.device("cuda")
            >>> val_metrics = self._validate(model, val_loader, device)
            >>> print(f"Validation Loss: {val_metrics['loss']:.4f}")
            >>> print(f"Validation Accuracy: {val_metrics['accuracy']:.2%}")
        """

        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return {"loss": total_loss / len(val_loader), "accuracy": correct / total}

    @abstractmethod
    def _prepare_model_and_optimizer(
        self, model: nn.Module, config: dict
    ) -> tuple[nn.Module, torch.optim.Optimizer]:
        """Abstract method to prepare model and optimizer for the specific strategy.

        This is the key customization point for each fine-tuning strategy. Concrete
        implementations should configure which parameters are trainable and create
        an appropriate optimizer. This method acts as a "hook" in the Template Method
        pattern.

        Args:
            model (nn.Module): The PyTorch model to be configured for fine-tuning.
                May be a pre-trained model that needs specific layers frozen/unfrozen.
            config (dict): Configuration dictionary containing training parameters.
                Expected to have a 'training' key with nested 'learning_rate' and
                other hyperparameters.

        Returns:
            tuple[nn.Module, torch.optim.Optimizer]: A tuple containing:
                - The configured model with appropriate layers frozen/unfrozen
                - An optimizer instance configured with the trainable parameters

        Raises:
            NotImplementedError: This method must be implemented by concrete subclasses.

        Note:
            Common patterns in implementations include:
            - Freezing all layers except the final classifier (LastLayerStrategy)
            - Unfreezing all layers (full fine-tuning)
            - Gradually unfreezing layers (progressive fine-tuning)
            - Using different learning rates for different layer groups

        Example:
            >>> def _prepare_model_and_optimizer(self, model, config):
            ...     # Freeze all parameters
            ...     for param in model.parameters():
            ...         param.requires_grad = False
            ...     # Unfreeze final layer
            ...     for param in model.fc.parameters():
            ...         param.requires_grad = True
            ...     # Create optimizer with unfrozen parameters
            ...     optimizer = torch.optim.Adam(
            ...         filter(lambda p: p.requires_grad, model.parameters()),
            ...         lr=config['training']['learning_rate']
            ...     )
            ...     return model, optimizer
        """
        pass

    def execute(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
    ) -> dict[str, float]:
        """Execute the complete fine-tuning strategy (Template Method).

        This is the main public interface for running a fine-tuning strategy. It
        orchestrates the entire training pipeline including model preparation,
        training loop execution, and validation. This method defines the algorithm
        structure while allowing customization through the abstract methods.

        Args:
            model (nn.Module): The PyTorch model to fine-tune. Should already have
                the correct output dimension for the target task.
            train_loader (DataLoader): DataLoader for training data. Should provide
                batches of (inputs, labels) tuples.
            val_loader (DataLoader): DataLoader for validation data. Used to evaluate
                model performance after each epoch.
            config (dict): Configuration dictionary containing all training parameters.
                Expected structure:
                {
                    'training': {
                        'epochs': int,  # Number of training epochs
                        'learning_rate': float,  # Learning rate for optimizer
                        # ... other training parameters
                    }
                }

        Returns:
            dict: Dictionary containing the best validation metrics achieved during
                training. Keys include:
                - 'loss' (float): Best validation loss
                - 'accuracy' (float): Best validation accuracy (0-1 scale)

        Note:
            - The model is automatically moved to GPU if available
            - Best model selection is based on validation accuracy
            - Progress is printed for each epoch
            - This method can be overridden by subclasses for custom behavior
              (see NoTrainingStrategy for an example)

        Example:
            >>> config = {
            ...     'training': {
            ...         'epochs': 10,
            ...         'learning_rate': 0.001
            ...     }
            ... }
            >>> strategy = LastLayerStrategy()
            >>> metrics = strategy.execute(model, train_loader, val_loader, config)
            >>> print(f"Best validation accuracy: {metrics['accuracy']:.2%}")
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Store original device to restore
        original_device = next(model.parameters()).device
        model = model.to(device)

        model, optimizer = self._prepare_model_and_optimizer(model, config)

        epochs = config["training"]["epochs"]
        best_val_metrics = {"loss": float("inf"), "accuracy": 0.0}

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            train_loss = self._train_one_epoch(model, train_loader, optimizer, device)
            val_metrics = self._validate(model, val_loader, device)

            print(
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Accuracy: {val_metrics['accuracy']:.4f}"
            )

            if val_metrics["accuracy"] > best_val_metrics["accuracy"]:
                best_val_metrics = val_metrics

        # Move model back (test problems)
        model = model.to(original_device)

        return best_val_metrics


class LastLayerStrategy(FineTuningStrategy):
    """Fine-tuning strategy that only trains the final classification layer.

    This strategy implements transfer learning by freezing all layers except the
    final classifier layer. This approach is computationally efficient and often
    effective when the pre-trained features are already well-suited to the target
    task. It's particularly useful when the target dataset is small or similar to
    the original training data.

    The strategy works by:
    1. Freezing all parameters in the model (requires_grad = False)
    2. Unfreezing only the final classification layer
    3. Creating an optimizer that only updates the unfrozen parameters

    Attributes:
        Inherits all attributes from FineTuningStrategy.

    Note:
        This strategy is ideal for:
        - Small datasets where overfitting is a concern
        - Tasks similar to the original pre-training task
        - Quick experimentation and baseline establishment
        - Limited computational resources

    Example:
        >>> strategy = LastLayerStrategy()
        >>> model = torchvision.models.resnet18(pretrained=True)
        >>> # Assume model's final layer has been replaced for target task
        >>> metrics = strategy.execute(model, train_loader, val_loader, config)
        >>> print(f"Final layer only - Accuracy: {metrics['accuracy']:.2%}")
    """

    def _prepare_model_and_optimizer(
        self, model: nn.Module, config: dict
    ) -> tuple[nn.Module, torch.optim.Optimizer]:
        """Prepare model by freezing all layers except the final classifier.

        This method configures the model for last-layer-only fine-tuning by setting
        requires_grad=False for all parameters except those in the final layer.
        It uses the get_final_layer utility to dynamically identify the last layer
        regardless of model architecture.

        Args:
            model (nn.Module): Pre-trained model to be configured. Can be any
                architecture (ResNet, VGG, etc.) as long as it has a final
                Linear layer.
            config (dict): Configuration dictionary containing training parameters.
                Expected to have 'training.learning_rate' for optimizer setup.

        Returns:
            tuple[nn.Module, torch.optim.Optimizer]: A tuple containing:
                - model: The input model with all layers frozen except the final one
                - optimizer: Adam optimizer configured with only the unfrozen
                  parameters and the specified learning rate

        Note:
            The method prints information about which layer was unfrozen for
            transparency. Only parameters with requires_grad=True are included
            in the optimizer to avoid unnecessary memory usage.

        Example:
            >>> config = {'training': {'learning_rate': 0.001}}
            >>> model, optimizer = strategy._prepare_model_and_optimizer(model, config)
            >>> # Now only the final layer parameters will be updated during training
        """

        for param in model.parameters():
            param.requires_grad = False

        final_layer_name, final_layer = get_final_layer(model)

        for param in final_layer.parameters():
            param.requires_grad = True

        print(f"Unfrozen final layer: {final_layer_name}")

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["training"]["learning_rate"],
        )

        return model, optimizer


class FeatureExtractionStrategy(FineTuningStrategy):
    """Strategy using the model as a fixed feature extractor
      with a classical ML classifier.

    This strategy treats the pre-trained model as a fixed feature extractor by
    replacing the final classification layer with an Identity layer. Features are
    extracted for all training and validation samples, then a classical machine
    learning algorithm (Logistic Regression by default) is trained on these features.

    This approach is particularly effective when:
    - The pre-trained features are highly relevant to the target task
    - You want to leverage non-neural network classifiers
    - Training data is limited and you want to avoid overfitting
    - You need interpretable classifiers or probability calibration

    The strategy follows these steps:
    1. Replace the final layer with nn.Identity() to extract features
    2. Extract feature vectors for all training samples
    3. Train a Logistic Regression classifier on extracted features
    4. Evaluate the classifier on validation features

    Attributes:
        Inherits all attributes from FineTuningStrategy.

    Note:
        Unlike other strategies, this doesn't update any neural network parameters.
        The neural network is used purely as a feature extractor, and all learning
        happens in the classical ML classifier.

    Example:
        >>> strategy = FeatureExtractionStrategy()
        >>> model = torchvision.models.resnet18(pretrained=True)
        >>> # Model's final layer will be replaced with Identity
        >>> metrics = strategy.execute(model, train_loader, val_loader, config)
        >>> print(f"Logistic Regression on features - "
        >>> f"Accuracy: {metrics['accuracy']:.2%}")
    """

    def _prepare_model_and_optimizer(
        self, model: nn.Module, config: dict
    ) -> tuple[nn.Module, torch.optim.Optimizer]:
        """Return model unchanged with dummy optimizer (not used in this strategy).

        Since this strategy doesn't use gradient-based optimization, this method
        returns a dummy optimizer. The actual model preparation (replacing final
        layer with Identity) happens in the execute() method.

        Args:
            model (nn.Module): The model (returned unchanged at this stage).
            config (dict): Configuration dictionary (not used here).

        Returns:
            tuple[nn.Module, torch.optim.Optimizer]: A tuple containing:
                - model: The unchanged input model
                - optimizer: A dummy SGD optimizer with learning rate 0

        Note:
            The actual model modification happens in execute() where the final
            layer is replaced with nn.Identity().
        """

        optimizer = torch.optim.SGD(model.parameters(), lr=0)

        return model, optimizer

    def extract_features(
        self, model: nn.Module, data_loader: DataLoader, device: torch.device
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract feature vectors and labels from a dataset using the model.

        Passes all data through the model (which should have nn.Identity as its
        final layer) to extract feature representations. Features are collected
        and returned as numpy arrays for use with scikit-learn classifiers.

        Args:
            model (nn.Module): Feature extraction model with Identity final layer.
                Should be in evaluation mode and on the correct device.
            data_loader (DataLoader): DataLoader providing batches of data to
                process. Should yield (inputs, labels) tuples.
            device (torch.device): Device where the model and computations are
                performed (CPU or CUDA).

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - features: 2D array of shape (n_samples, n_features) containing
                  extracted feature vectors
                - labels: 1D array of shape (n_samples,) containing corresponding
                  class labels

        Note:
            - Model is set to eval mode to ensure consistent feature extraction
            - No gradients are computed to save memory
            - Features are moved to CPU before converting to numpy
            - Progress is displayed using tqdm

        Example:
            >>> model.fc = nn.Identity()  # Replace final layer
            >>> features, labels = self._extract_features(model, loader, device)
            >>> print(f"Extracted features shape: {features.shape}")
        """

        model.eval()
        all_features = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc="Extracting features"):
                inputs = inputs.to(device)
                features = model(inputs)

                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())

        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        return features, labels

    def execute(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
    ) -> dict:
        """Execute feature extraction and train a Logistic Regression classifier.

        Overrides the base execute() to implement the feature extraction strategy.
        The pre-trained model is converted to a feature extractor, features are
        extracted from both training and validation sets, and a Logistic Regression
        classifier is trained on these features.

        Args:
            model (nn.Module): Pre-trained model to use as feature extractor.
                Its final layer will be replaced with nn.Identity().
            train_loader (DataLoader): DataLoader for training data from which
                features will be extracted to train the classifier.
            val_loader (DataLoader): DataLoader for validation data used to
                evaluate the trained classifier.
            config (dict): Configuration dictionary (not used in current
                implementation but maintained for interface compatibility).

        Returns:
            dict: Dictionary containing evaluation metrics:
                - 'accuracy' (float): Validation accuracy of the Logistic
                  Regression classifier (0-1 scale)

        Note:
            - The model's final layer is permanently replaced with nn.Identity()
            - Logistic Regression uses max_iter=1000 for convergence
            - All computations use the best available device (GPU if available)
            - Feature extraction progress is displayed with tqdm

        Example:
            >>> strategy = FeatureExtractionStrategy()
            >>> model = torchvision.models.vgg16(pretrained=True)
            >>> metrics = strategy.execute(model, train_loader, val_loader, config)
            >>> print(f"Feature extraction + LR accuracy: {metrics['accuracy']:.2%}")

        Raises:
            sklearn.exceptions.ConvergenceWarning: If Logistic Regression doesn't
                converge within max_iter iterations (suppressed by max_iter=1000).
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Store original device to restore
        original_device = next(model.parameters()).device
        model = model.to(device)

        final_layer_name, _ = get_final_layer(model)

        set_nested_attr(model, final_layer_name, nn.Identity())

        train_features, train_labels = self.extract_features(
            model, train_loader, device
        )
        val_features, val_labels = self.extract_features(model, val_loader, device)

        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(train_features, train_labels)

        val_predictions = classifier.predict(val_features)
        accuracy = np.mean(val_predictions == val_labels)

        # Move model back (test problems)
        model = model.to(original_device)

        return {"accuracy": accuracy}


class FullFineTuningStrategy(FineTuningStrategy):
    """Fine-tuning strategy that trains all layers in the model.

    This strategy implements full fine-tuning by unfreezing all parameters in the
    model, allowing the entire network to be updated during training. This approach
    can achieve the highest performance when sufficient training data is available,
    but requires more computational resources.

    The strategy works by:
    1. Ensuring all parameters have requires_grad = True
    2. Creating an optimizer that updates all model parameters
    3. Training the entire network end-to-end

    This approach is ideal for:
    - Large datasets where overfitting is less of a concern
    - Tasks that differ significantly from the original pre-training task
    - When maximum performance is desired and computational resources allow
    - Fine-grained adaptation of low-level features

    Attributes:
        Inherits all attributes from FineTuningStrategy.

    Note:
        Full fine-tuning typically requires:
        - Lower learning rates to avoid destroying pre-trained features
        - More careful regularization and hyperparameter tuning
        - Sufficient training data to avoid overfitting
        - More computational resources and training time

    Example:
        >>> strategy = FullFineTuningStrategy()
        >>> model = torchvision.models.resnet18(pretrained=True)
        >>> # Assume model's final layer has been replaced for target task
        >>> metrics = strategy.execute(model, train_loader, val_loader, config)
        >>> print(f"Full fine-tuning - Accuracy: {metrics['accuracy']:.2%}")
    """

    def _prepare_model_and_optimizer(
        self, model: nn.Module, config: dict
    ) -> tuple[nn.Module, torch.optim.Optimizer]:
        """Prepare model by ensuring all layers are trainable.

        This method configures the model for full fine-tuning by setting
        requires_grad=True for all parameters in the model. This allows the
        entire network to be updated during training.

        Args:
            model (nn.Module): Pre-trained model to be configured. All layers
                will be made trainable regardless of their current state.
            config (dict): Configuration dictionary containing training parameters.
                Expected to have 'training.learning_rate' for optimizer setup.

        Returns:
            tuple[nn.Module, torch.optim.Optimizer]: A tuple containing:
                - model: The input model with all layers unfrozen for training
                - optimizer: Adam optimizer configured with all model parameters
                  and the specified learning rate

        Note:
            The method prints the total number of trainable parameters for
            transparency. All model parameters are included in the optimizer.

        Example:
            >>> config = {'training':
              {'learning_rate': 0.0001}}  # Lower LR for full fine-tuning
            >>> model, optimizer = strategy._prepare_model_and_optimizer(model, config)
            >>> # Now all model parameters will be updated during training
        """

        for param in model.parameters():
            param.requires_grad = True

        print(
            "All layers unfrozen. Total trainable params: "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["training"]["learning_rate"]
        )

        return model, optimizer

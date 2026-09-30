from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class DataManager:
    """Manages data loading flexibly using an internal configuration registry.

    This class provides a flexible way to load different datasets by maintaining
    a configuration registry for supported datasets and their parameters. It handles
    dataset instantiation, transformation pipeline creation, and DataLoader setup
    for both training and validation data.

    Attributes:
        DATASET_CONFIGS (dict): A dictionary mapping dataset names to their
        configuration dictionaries. Each configuration contains the dataset class,
        number of classes, and arguments for training and validation splits.
        config (dict): Configuration dictionary containing data specifications passed
        during initialization.
        dataset_name (str): Name of the selected dataset from the configuration.
        dataset_config (dict): Configuration dictionary for the specific dataset being
        used.
        train_dataset (torch.utils.data.Dataset or None): Training dataset instance.
        val_dataset (torch.utils.data.Dataset or None): Validation dataset instance.
        num_classes (int): Number of classes in the dataset.
        class_names (list[str]): List of class names from the dataset.
        num_workers (int): Number of workers for data loading processes.

    Example:
        >>> config = {
        ...     "data": {
        ...         "name": "CIFAR10",
        ...         "path": "./data",
        ...         "batch_size": 32
        ...     }
        ... }
        >>> data_manager = DataManager(config)
        >>> data_manager.load_data()
        >>> train_loader = data_manager.get_train_loader()
    """

    # TODO: Complete the dataset configurations based on each dataset's properties
    # (number of classes, splits, target types, etc.)
    # Note: PCAM has to be downloaded manually.

    DATASET_CONFIGS = {
        "CIFAR10": {
            "dataset": datasets.CIFAR10,
            "num_classes": 10,
            "batch_size": 4,
            "train": True,
            "root": "data",
            "download": True,
        },
        "OxfordIIITPet": {
            "dataset": datasets.OxfordIIITPet,
            "num_classes": 2,
            "batch_size": 8,
            "split": {"train": "trainval", "val": "test"},
            "target_types": ["binary-category"],
            "root": "data",
            "download": True,
        },
        "PCAM": {
            "dataset": datasets.PCAM,
            "num_classes": 2,
            "batch_size": 16,
            "split": {"train": "train", "val": "val"},
            "root": "data",
            "download": True,
        },
    }

    def __init__(self, config: dict) -> None:
        """Initialize the DataManager with configuration settings.

        Validates the provided configuration to ensure a supported dataset is specified,
        then caches the dataset configuration and initializes instance attributes.

        Args:
            config (dict): Configuration dictionary that must contain a 'data' key
                with 'name', 'path', and 'batch_size' subkeys. The dataset name
                must be present in the DATASET_CONFIGS registry.

        Raises:
            ValueError: If no dataset name is specified in config['data']['name'].
            ValueError: If the specified dataset is not supported (not found in
                DATASET_CONFIGS registry).

        Example:
            >>> config = {
            ...     "data": {
            ...         "name": "CIFAR10",
            ...         "path": "./datasets",
            ...         "batch_size": 64
            ...     }
            ... }
            >>> data_manager = DataManager(config)
        """
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.class_names: list[str] = []
        self.num_workers = 2

        # Validate and cache dataset configuration early
        data_config = self.config.get("data", {})
        dataset_name = data_config.get("name")

        if not dataset_name:
            raise ValueError("Dataset name must be specified in config['data']['name']")

        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(
                f"Dataset '{dataset_name}' not supported."
                f"Options: {list(self.DATASET_CONFIGS.keys())}"
            )

        # Cache the dataset configuration for later use
        self.dataset_name = dataset_name
        self.dataset_config = self.DATASET_CONFIGS[dataset_name]
        self.num_classes = self.dataset_config["num_classes"]

    def _get_transforms(self) -> transforms.Compose:
        """Create and return a standardized data transformation pipeline.

        Creates a transforms.Compose pipeline optimized for transfer learning with
        ImageNet pre-trained models. The pipeline includes image resizing to ImageNet
        dimensions, tensor conversion, and normalization using ImageNet statistics.

        Returns:
            transforms.Compose: The composed transformation pipeline that includes:
                - Resize to images for ImageNet compatibility
                - Convert PIL images to tensors
                - Normalize with ImageNet statistics

        Note:
            These transformations are essential for transfer learning as they ensure
            input data matches the preprocessing used during ImageNet pre-training.

        Example:
            >>> data_manager = DataManager(config)
            >>> transforms = data_manager._get_transforms()
            >>> # transforms can now be applied to PIL images
        """
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load_train_data(self) -> None:
        """Load the training dataset using the dataset configuration.

        Uses the dataset class and parameters from the configuration registry
        to instantiate the training dataset. Applies the transformations from
        _get_transforms() and updates class information.

        Updates:
            self.train_dataset: The loaded training dataset instance
            self.class_names: List of class names from the dataset
        """
        common_args = {
            key: value
            for key, value in self.dataset_config.items()
            if key not in ["dataset", "num_classes", "split", "train", "batch_size"]
        }

        # Different dataset patterns
        if "split" in self.dataset_config.keys():
            self.train_dataset = self.dataset_config["dataset"](
                **common_args,
                split=self.dataset_config["split"]["train"],
                transform=self._get_transforms(),
            )

        else:
            self.train_dataset = self.dataset_config["dataset"](
                **common_args,
                train=self.dataset_config["train"],
                transform=self._get_transforms(),
            )

        if self.train_dataset is not None:
            try:
                self.class_names = self.train_dataset.classes

            except Exception:
                self.class_names = [str(i) for i in range(self.num_classes)]

    def load_val_data(self) -> None:
        """Load the training dataset using the dataset configuration.

        Uses the dataset class and parameters from the configuration registry
        to instantiate the validation dataset. Applies the transformations from
        _get_transforms() and updates class information.

        Updates:
            self.val_dataset: The loaded training dataset instance
            self.class_names: List of class names from the dataset
        """
        common_args = {
            key: value
            for key, value in self.dataset_config.items()
            if key not in ["dataset", "num_classes", "split", "train", "batch_size"]
        }

        # Different dataset patterns
        if "split" in self.dataset_config.keys():
            self.val_dataset = self.dataset_config["dataset"](
                **common_args,
                split=self.dataset_config["split"]["val"],
                transform=self._get_transforms(),
            )

        else:
            self.val_dataset = self.dataset_config["dataset"](
                **common_args,
                train=not self.dataset_config["train"],
                transform=self._get_transforms(),
            )

        if self.val_dataset is not None:
            try:
                self.class_names = self.val_dataset.classes

            except Exception:
                self.class_names = [str(i) for i in range(self.num_classes)]

    def load_data(self) -> None:
        """Loads both training and validation datasets.

        Convenience method that loads both training and validation data.
        """
        self.load_train_data()
        self.load_val_data()

    def get_train_loader(self) -> DataLoader:
        """Return a DataLoader for the training dataset.

        Creates a DataLoader with appropriate batch size and shuffling settings
        for training. Ensures the training dataset has been loaded before creating
        the loader.

        Returns:
            DataLoader: Training data loader configured for training

        Raises:
            ValueError: If training dataset has not been loaded yet
        """

        if self.train_dataset is None:
            raise ValueError("Training dataset has not been loaded yet")

        return DataLoader(
            self.train_dataset,
            batch_size=self.dataset_config["batch_size"],
            shuffle=True,
        )

    def get_val_loader(self) -> DataLoader:
        """Return a DataLoader for the training dataset.

        Creates a DataLoader with appropriate batch size and shuffling settings
        for validation. Ensures the validation dataset has been loaded before creating
        the loader.

        Returns:
            DataLoader: Validation data loader configured for validation

        Raises:
            ValueError: If validation dataset has not been loaded yet
        """
        if self.val_dataset is None:
            raise ValueError("Validation dataset has not been loaded yet")

        return DataLoader(
            self.val_dataset,
            batch_size=self.dataset_config["batch_size"],
            shuffle=False,
        )

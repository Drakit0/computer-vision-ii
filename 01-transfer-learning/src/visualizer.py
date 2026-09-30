# src/visualizer.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader


from src.data_manager import DataManager
from src.model_handler import ModelHandler
from src.strategies import FeatureExtractionStrategy


class FeatureVisualizer:
    """
    A class to visualize high-dimensional features using PCA.
    """

    def __init__(self, config: dict) -> None:
        """
        Initializes the FeatureVisualizer with a configuration.

        Args:
            config (dict): A configuration dictionary containing data, model,
            and training settings.
        """
        self.config = config
        self.data_manager = DataManager(config)
        self.model_handler = ModelHandler(config)

    def visualize_features(self) -> None:
        """
        Runs the feature extraction, PCA, and visualization pipeline.
        """
        self.data_manager.load_data()
        train_loader = self.data_manager.get_train_loader()

        self._extract_features_and_plot(train_loader)

    def _extract_features_and_plot(self, train_loader: DataLoader) -> None:
        """
        Extract features and calls _plot_pca to represent them.

        Args:
            features_2d (np.ndarray): The 2D array of features after PCA.
            labels (np.ndarray): The corresponding labels for the features.
        """

        self.model_handler.load_model()
        model = self.model_handler.model
        strategy = FeatureExtractionStrategy()

        if model is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            features, labels = strategy.extract_features(model, train_loader, device)

            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)

            self._plot_pca(features_2d, labels)

    def _plot_pca(self, features_2d: np.ndarray, labels: np.ndarray) -> None:
        """
        Plots the 2D PCA-transformed features.

        Args:
            features_2d (np.ndarray): The 2D array of features after PCA.
            labels (np.ndarray): The corresponding labels for the features.
        """

        colors = ["red" if label == 0 else "green" for label in labels]

        plt.figure(figsize=(10, 8))
        plt.scatter(features_2d[:, 0], features_2d[:, 1], c=colors)
        plt.scatter([], [], c="red", label="Class 0")
        plt.scatter([], [], c="green", label="Class 1")
        plt.legend()
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Visualization of Features")
        plt.show()

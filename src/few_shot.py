import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets


class FeatureExtractor:
    """
    Extracts features from different model architectures.
    Handles VGG16, ResNet18, and EfficientNet_B0.
    Uses hooks to capture activations from the penultimate layer (before final classification).
    """
    
    def __init__(self, model: nn.Module, model_name: str, device: torch.device):
        """
        Args:
            model: Pre-trained model (ImageNet weights) for feature extraction
            model_name: Name of the model architecture ('vgg16', 'resnet18', 'efficientnet_b0')
                       Used to identify which layer to hook for feature extraction
            device: torch device
        """
        self.model = model.to(device)
        self.model_name = model_name.lower()
        self.device = device
        self.model.eval()
        
        # Register hook to extract features
        self.features = None
        self._register_hook()
    
    def _register_hook(self):
        """Register a hook to capture features from the penultimate layer"""
        if self.model_name == "vgg16":
            # Hook into the first linear layer of classifier
            self.model.classifier[6].register_forward_hook(self._hook_fn)
        elif self.model_name == "resnet18":
            # Hook into the final FC layer
            self.model.fc.register_forward_hook(self._hook_fn)
        elif self.model_name == "efficientnet_b0":
            # Hook into the final linear layer of classifier
            self.model.classifier[1].register_forward_hook(self._hook_fn)
        else:
            raise ValueError(f"Model {self.model_name} not supported")
    
    def _hook_fn(self, module, input, output):
        """Hook function to capture input to the final layer (features)"""
        self.features = input[0].detach()
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.
        
        Args:
            images: Batch of images (B, C, H, W)
            
        Returns:
            Feature vectors (B, feature_dim)
        """
        with torch.no_grad():
            _ = self.model(images.to(self.device))
        return self.features.cpu()


class SimpleShot:
    """
    SimpleShot: A simple few-shot learning method based on nearest-centroid classification.
    Extracts features using a pre-trained model and classifies based on distance to class centroids.
    """
    
    def __init__(self, model: nn.Module, model_name: str, device: torch.device):
        """
        Args:
            model: Pre-trained model (ImageNet weights) for feature extraction
            model_name: Model architecture name ('vgg16', 'resnet18', 'efficientnet_b0')
                       Required to identify the correct layer for feature extraction,
                       as different architectures have different layer naming conventions
            device: torch device
        """
        self.feature_extractor = FeatureExtractor(model, model_name, device)
        self.device = device
        self.centroids = None
        self.classes = None
        self.feature_mean = None
    
    def fit(self, support_images: torch.Tensor, support_labels: torch.Tensor):
        """
        Compute class centroids from support set.
        
        Args:
            support_images: Support set images (N, C, H, W)
            support_labels: Support set labels (N,)
        """
        # Extract features from support set
        support_features = self.feature_extractor.extract_features(support_images)
        
        # --- SimpleShot Transformations ---
        # 1. Centering: Subtract the mean of the support set features
        self.feature_mean = support_features.mean(dim=0, keepdim=True)
        support_features = support_features - self.feature_mean
        # 2. L2 Normalization: Normalize feature vectors to unit length
        support_features = torch.nn.functional.normalize(support_features, p=2, dim=1)
        
        # Get unique classes
        self.classes = torch.unique(support_labels)
        
        # Compute centroids for each class
        self.centroids = {}
        for cls in self.classes:
            mask = support_labels == cls
            class_features = support_features[mask]
            centroid = class_features.mean(dim=0)
            self.centroids[cls.item()] = centroid
    
    def predict(self, query_images: torch.Tensor) -> torch.Tensor:
        """
        Classify query images using nearest-centroid matching.
        
        Args:
            query_images: Query set images (M, C, H, W)
            
        Returns:
            Predicted class labels (M,)
        """
        if self.centroids is None:
            raise RuntimeError("Must call fit() first to compute class centroids")
        
        # Extract features from query set
        query_features = self.feature_extractor.extract_features(query_images)
        
        # --- SimpleShot Transformations ---
        # 1. Centering: Subtract the previously computed support set mean
        query_features = query_features - self.feature_mean
        # 2. L2 Normalization: Normalize query feature vectors to unit length
        query_features = torch.nn.functional.normalize(query_features, p=2, dim=1)
        
        # Compute distances to all centroids (using Euclidean distance)
        predictions = []
        for query_feature in query_features:
            distances = {}
            for cls, centroid in self.centroids.items():
                dist = torch.norm(query_feature - centroid, p=2)
                distances[cls] = dist
            
            # Predict class with minimum distance
            predicted_cls = min(distances, key=distances.get)
            predictions.append(predicted_cls)
        
        return torch.tensor(predictions)


def create_few_shot_episode(dataset: datasets.ImageFolder, n_way: int, k_shot: int, 
                           q_shot: int = 1, seed: int = None) -> Tuple[torch.Tensor, torch.Tensor, 
                                                       torch.Tensor, torch.Tensor]:
    """
    Create a few-shot episode (N-way K-shot task).
    
    Args:
        dataset: ImageFolder dataset
        n_way: Number of classes
        k_shot: Number of support samples per class
        q_shot: Number of query samples per class (default: 1)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (support_images, support_labels, query_images, query_labels)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get all class indices
    class_to_indices = {}
    for idx, label in enumerate(dataset.targets):
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)
    
    # Sample n_way classes
    total_per_class = k_shot + q_shot
    available_classes = [cls for cls in class_to_indices.keys() 
                        if len(class_to_indices[cls]) >= total_per_class]
    
    if len(available_classes) < n_way:
        raise ValueError(f"Not enough classes with >= {total_per_class} samples. "
                        f"Available: {len(available_classes)}, Required: {n_way}")
    
    selected_classes = np.random.choice(available_classes, size=n_way, replace=False)
    
    support_images = []
    support_labels = []
    query_images = []
    query_labels = []
    
    for new_label, cls in enumerate(selected_classes):
        indices = class_to_indices[cls]
        selected_indices = np.random.choice(indices, size=k_shot + q_shot, replace=False)
        
        support_idx = selected_indices[:k_shot]
        query_idx = selected_indices[k_shot:]
        
        for idx in support_idx:
            image, _ = dataset[idx]
            support_images.append(image)
            support_labels.append(new_label)
        
        for idx in query_idx:
            image, _ = dataset[idx]
            query_images.append(image)
            query_labels.append(new_label)
    
    support_images = torch.stack(support_images)
    support_labels = torch.tensor(support_labels)
    query_images = torch.stack(query_images)
    query_labels = torch.tensor(query_labels)
    
    return support_images, support_labels, query_images, query_labels


def evaluate_few_shot(model: nn.Module, model_name: str, dataset: datasets.ImageFolder,
                     n_way: int = 5, k_shot: int = 5, q_shot: int = 1, n_episodes: int = 100,
                     device: torch.device = None) -> Dict[str, float]:
    """
    Evaluate SimpleShot on few-shot learning task.
    
    Args:
        model: Pre-trained model
        model_name: Model architecture name
        dataset: ImageFolder dataset to sample episodes from
        n_way: Number of classes per episode
        k_shot: Number of support samples per class
        q_shot: Number of query samples per class (default: 1)
        n_episodes: Number of episodes to evaluate
        device: torch device
        
    Returns:
        Dictionary with evaluation metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    
    accuracies = []
    
    for episode in range(n_episodes):
        try:
            support_images, support_labels, query_images, query_labels = \
                create_few_shot_episode(dataset, n_way, k_shot, q_shot, seed=episode)
            
            # Create and fit SimpleShot model
            simple_shot = SimpleShot(model, model_name, device)
            simple_shot.fit(support_images, support_labels)
            
            # Predict on query set
            predictions = simple_shot.predict(query_images)
            
            # Compute accuracy
            accuracy = (predictions == query_labels).float().mean().item()
            accuracies.append(accuracy)
            
        except Exception as e:
            print(f"Warning: Episode {episode} failed with error: {e}")
            continue
    
    if not accuracies:
        raise RuntimeError("All episodes failed!")
    
    accuracies = np.array(accuracies)
    
    return {
        "mean_accuracy": accuracies.mean(),
        "std_accuracy": accuracies.std(),
        "n_episodes": len(accuracies),
        "n_way": n_way,
        "k_shot": k_shot,
        "q_shot": q_shot,
    }

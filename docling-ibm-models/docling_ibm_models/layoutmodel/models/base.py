#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
"""Base interfaces for layout detection models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from PIL import Image


@dataclass
class ModelConfig:
    """Configuration for layout detection model.
    
    Parameters
    ----------
    model_type : str
        Type of layout model to use (e.g., "docling", "nvidia_nemo", "custom")
    artifact_path : str
        Path to model artifacts directory
    device : str
        Device to run inference on ("cpu", "cuda", "mps")
    num_threads : int
        Number of threads for CPU inference
    threshold : float
        Confidence threshold for predictions
    blacklist_classes : Set[str]
        Classes to filter out from predictions
    requires_label_offset : Optional[bool]
        Whether model requires label offset (auto-detected if None)
    image_size : Optional[Tuple[int, int]]
        Target image size for preprocessing (auto-detected if None)
    model_specific_params : Dict[str, Any]
        Additional model-specific parameters
    """
    
    model_type: str = "docling"
    artifact_path: str = ""
    device: str = "cpu"
    num_threads: int = 4
    threshold: float = 0.3
    blacklist_classes: Set[str] = field(default_factory=set)
    requires_label_offset: Optional[bool] = None
    image_size: Optional[Tuple[int, int]] = None
    model_specific_params: Dict[str, Any] = field(default_factory=dict)


class BaseLayoutModel(ABC):
    """Abstract base class for layout detection models.
    
    Defines the interface for loading models, preprocessing images,
    running inference, and post-processing outputs. Different layout
    detection models (Docling, NVIDIA NeMo, etc.) implement this interface.
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize model with configuration.
        
        Parameters
        ----------
        config : ModelConfig
            Model configuration
        """
        self.config = config
        self._device = torch.device(config.device)
        
        # Set CPU threads if applicable
        if config.device == "cpu":
            torch.set_num_threads(config.num_threads)
    
    @abstractmethod
    def load_model(self) -> None:
        """Load model and processor from artifacts.
        
        Should populate internal model and processor attributes.
        Should validate that required files exist in artifact_path.
        
        Raises
        ------
        FileNotFoundError
            If required model files are missing
        """
        pass
    
    @abstractmethod
    def preprocess(
        self, images: List[Image.Image]
    ) -> Tuple[Any, torch.Tensor]:
        """Preprocess images for model inference.
        
        Parameters
        ----------
        images : List[Image.Image]
            List of PIL images in RGB format
        
        Returns
        -------
        Tuple[Any, torch.Tensor]
            Preprocessed inputs ready for model and target sizes tensor
        """
        pass
    
    @abstractmethod
    def infer(self, inputs: Any) -> Any:
        """Run model inference.
        
        Parameters
        ----------
        inputs : Any
            Preprocessed inputs from preprocess()
        
        Returns
        -------
        Any
            Raw model outputs
        """
        pass
    
    @abstractmethod
    def postprocess(
        self, outputs: Any, target_sizes: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """Post-process model outputs to detection format.
        
        Parameters
        ----------
        outputs : Any
            Raw model outputs from infer()
        target_sizes : torch.Tensor
            Target sizes for rescaling boxes
        
        Returns
        -------
        List[Dict[str, torch.Tensor]]
            List of detection dicts, each containing:
            - "scores": Tensor of confidence scores
            - "labels": Tensor of class label IDs
            - "boxes": Tensor of bounding boxes [x1, y1, x2, y2]
        """
        pass
    
    @abstractmethod
    def get_label_offset(self) -> int:
        """Get label offset for this model architecture.
        
        Returns
        -------
        int
            Label offset to add to predicted label IDs
        """
        pass
    
    @abstractmethod
    def info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns
        -------
        Dict[str, Any]
            Information about model configuration
        """
        pass

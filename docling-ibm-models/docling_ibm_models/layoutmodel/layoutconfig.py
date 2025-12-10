#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
"""
Simplified configuration management for layout models.

All configuration is managed through a simple nested dictionary structure.
ModelConfig dataclass is created from the merged config dict.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple

_log = logging.getLogger(__name__)


# Complete configuration hierarchy
LAYOUT_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Base defaults - applied to all models
    "defaults": {
        "device": "cpu",
        "num_threads": 4,
        "threshold": 0.3,
        "blacklist_classes": set(),
        # Model architecture settings (auto-detected if not specified)
        "requires_label_offset": None,
        "image_size": None,
    },
    
    # Model-specific configurations
    "models": {
        "docling": {
            "threshold": 0.3,
            "num_threads": 4,
        },
        "nvidia_nemo": {
            "threshold": 0.4,
            "num_threads": 8,
        },
        # Add more models here as needed
    }
}


@dataclass
class ModelConfig:
    """Configuration for layout detection model.
    
    This dataclass is created from the merged configuration dict.
    It provides a typed interface for model initialization.
    """
    model_type: str
    artifact_path: str
    device: str = "cpu"
    num_threads: int = 4
    threshold: float = 0.3
    blacklist_classes: Set[str] = field(default_factory=set)
    requires_label_offset: Optional[bool] = None
    image_size: Optional[Tuple[int, int]] = None
    
    @classmethod
    def from_dict(cls, model_type: str, artifact_path: str, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from a configuration dictionary.
        
        Parameters
        ----------
        model_type : str
            Type of model (e.g., "docling", "nvidia_nemo")
        artifact_path : str
            Path to model artifacts
        config_dict : Dict[str, Any]
            Merged configuration dictionary
            
        Returns
        -------
        ModelConfig
            Configuration instance
        """
        return cls(
            model_type=model_type,
            artifact_path=artifact_path,
            device=config_dict.get("device", "cpu"),
            num_threads=config_dict.get("num_threads", 4),
            threshold=config_dict.get("threshold", 0.3),
            blacklist_classes=config_dict.get("blacklist_classes", set()),
            requires_label_offset=config_dict.get("requires_label_offset"),
            image_size=config_dict.get("image_size"),
        )


def get_model_config(
    model_type: str,
    user_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get merged configuration for a model type.
    
    Priority: user_config > model-specific > defaults
    
    Parameters
    ----------
    model_type : str
        Type of layout model (e.g., "docling", "nvidia_nemo")
    user_config : Optional[Dict[str, Any]]
        User-provided configuration overrides
        
    Returns
    -------
    Dict[str, Any]
        Merged configuration dictionary
        
    Examples
    --------
    >>> get_model_config("nvidia_nemo")
    {'device': 'cpu', 'num_threads': 8, 'threshold': 0.4, ...}
    
    >>> get_model_config("nvidia_nemo", {"threshold": 0.6, "device": "cuda"})
    {'device': 'cuda', 'num_threads': 8, 'threshold': 0.6, ...}
    """
    # Start with defaults
    config = LAYOUT_MODEL_CONFIGS["defaults"].copy()
    
    # Apply model-specific config
    model_specific = LAYOUT_MODEL_CONFIGS["models"].get(model_type, {})
    config.update(model_specific)
    
    # Apply user config
    if user_config:
        config.update(user_config)
    
    _log.debug(f"Merged config for '{model_type}': {config}")
    
    return config


def create_model_config(
    model_type: str,
    artifact_path: str,
    user_config: Optional[Dict[str, Any]] = None,
) -> ModelConfig:
    """Create a ModelConfig instance with merged configuration.
    
    This is the main entry point for creating model configurations.
    
    Parameters
    ----------
    model_type : str
        Type of layout model
    artifact_path : str
        Path to model artifacts
    user_config : Optional[Dict[str, Any]]
        User-provided configuration overrides
        
    Returns
    -------
    ModelConfig
        Configured model instance
        
    Examples
    --------
    >>> config = create_model_config("nvidia_nemo", "/path/to/model")
    >>> config.threshold
    0.4
    
    >>> config = create_model_config("nvidia_nemo", "/path", {"threshold": 0.6})
    >>> config.threshold
    0.6
    """
    merged_config = get_model_config(model_type, user_config)
    return ModelConfig.from_dict(model_type, artifact_path, merged_config)

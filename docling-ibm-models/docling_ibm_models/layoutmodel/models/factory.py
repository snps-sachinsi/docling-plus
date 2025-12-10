#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
"""Factory for creating layout detection model instances."""

import logging
from typing import Callable, Dict, Optional, Type, Union, Any

from docling_ibm_models.layoutmodel.layoutconfig import ModelConfig, create_model_config
from docling_ibm_models.layoutmodel.models.base import BaseLayoutModel

_log = logging.getLogger(__name__)


class LayoutModelFactory:
    """Factory for creating layout detection model instances.
    
    Supports registration of different layout models (Docling, NVIDIA NeMo, etc.)
    and instantiation based on model type string. Uses lazy loading to avoid
    importing heavy dependencies until needed.
    """
    
    # Registry of model types - stores either class or callable that returns class
    _models: Dict[str, Union[Type[BaseLayoutModel], Callable]] = {}
    
    @classmethod
    def _get_default_models(cls) -> Dict[str, Callable]:
        """Get default model loaders (lazy loaded)."""
        def load_docling_model():
            from docling_ibm_models.layoutmodel.models.docling_model import (
                DoclingLayoutModel,
            )
            return DoclingLayoutModel
        
        def load_nvidia_model():
            from docling_ibm_models.layoutmodel.models.nvidia_model import (
                NvidiaLayoutModel,
            )
            return NvidiaLayoutModel
        
        def load_nvidia_page_elements_model():
            from docling_ibm_models.layoutmodel.models.nvidia_page_elements_model import (
                NvidiaPageElementsModel,
            )
            return NvidiaPageElementsModel
        
        return {
            "docling": load_docling_model,
            "nvidia_nemo": load_nvidia_model,
            "nvidia_page_elements": load_nvidia_page_elements_model,
        }
    
    @classmethod
    def register_model(
        cls, model_type: str, model_class: Type[BaseLayoutModel]
    ) -> None:
        """Register a new layout model type.
        
        Parameters
        ----------
        model_type : str
            Identifier for the model type (e.g., "nvidia_nemo", "custom")
        model_class : Type[BaseLayoutModel]
            Model class implementing BaseLayoutModel
        """
        cls._models[model_type] = model_class
        _log.info(f"Registered layout model type: {model_type}")
    
    @classmethod
    def create_model(
        cls,
        model_type: str = "docling",
        artifact_path: str = "",
        device: str = "cpu",
        num_threads: int = 4,
        threshold: float = 0.3,
        model_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> BaseLayoutModel:
        """Create a layout detection model instance.
        
        Parameters
        ----------
        model_type : str, optional
            Type of model to create, by default "docling"
        artifact_path : str, optional
            Path to model artifacts, by default ""
        device : str, optional
            Device for inference (overridden by model_config if provided)
        num_threads : int, optional
            Number of CPU threads (overridden by model_config if provided)
        threshold : float, optional
            Confidence threshold (overridden by model_config if provided)
        model_config : Optional[Dict[str, Any]], optional
            User configuration dict that merges with model defaults
        **kwargs
            Additional legacy parameters
        
        Returns
        -------
        BaseLayoutModel
            Instantiated layout model
        
        Raises
        ------
        ValueError
            If model_type is not registered
        """
        # Merge default models with registered ones
        all_models = {**cls._get_default_models(), **cls._models}
        
        if model_type not in all_models:
            available = ", ".join(all_models.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available models: {available}"
            )
        
        # If model_config is provided, use it; otherwise use explicit params
        if model_config is None:
            # Build config from explicit parameters
            model_config = {
                "device": device,
                "num_threads": num_threads,
                "threshold": threshold,
            }
        
        # Create configuration using the simplified API
        config = create_model_config(
            model_type=model_type,
            artifact_path=artifact_path,
            user_config=model_config,
        )
        
        # Get model class (may be callable for lazy loading)
        model_class_or_loader = all_models[model_type]
        if callable(model_class_or_loader) and not isinstance(model_class_or_loader, type):
            # It's a loader function, call it to get the class
            model_class = model_class_or_loader()
        else:
            model_class = model_class_or_loader
        
        # Instantiate model
        model = model_class(config)
        
        _log.info(f"Created {model_type} layout model: {model.info()}")
        return model
    
    @classmethod
    def list_models(cls) -> list:
        """List all registered model types.
        
        Returns
        -------
        list
            List of registered model type names
        """
        all_models = {**cls._get_default_models(), **cls._models}
        return list(all_models.keys())

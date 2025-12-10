#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
"""Docling layout detection model implementation."""

import logging
import os
import threading
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from transformers import AutoModelForObjectDetection, RTDetrImageProcessor

from docling_ibm_models.layoutmodel.layoutconfig import ModelConfig
from docling_ibm_models.layoutmodel.models.base import BaseLayoutModel

_log = logging.getLogger(__name__)

# Global lock for model initialization to prevent threading issues
_model_init_lock = threading.Lock()


class DoclingLayoutModel(BaseLayoutModel):
    """Docling layout detection model using HuggingFace Transformers.
    
    This is the default Docling layout model implementation using RTDetr
    or similar transformer-based object detection models.
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize Docling layout model.
        
        Parameters
        ----------
        config : ModelConfig
            Model configuration
        """
        super().__init__(config)
        self._model = None
        self._image_processor = None
        self._model_name = None
        self._label_offset = 0
        
        # Load model during initialization
        self.load_model()
    
    def load_model(self) -> None:
        """Load model and image processor from transformers.
        
        Raises
        ------
        FileNotFoundError
            If required configuration files are missing
        """
        artifact_path = self.config.artifact_path
        
        # Check for required files
        processor_config = os.path.join(artifact_path, "preprocessor_config.json")
        model_config = os.path.join(artifact_path, "config.json")
        st_file = os.path.join(artifact_path, "model.safetensors")
        
        if not os.path.isfile(st_file):
            raise FileNotFoundError(f"Missing safe tensors file: {st_file}")
        if not os.path.isfile(processor_config):
            raise FileNotFoundError(f"Missing processor config file: {processor_config}")
        if not os.path.isfile(model_config):
            raise FileNotFoundError(f"Missing model config file: {model_config}")
        
        # Load image processor
        self._image_processor = RTDetrImageProcessor.from_json_file(processor_config)
        
        # Load model with thread safety
        with _model_init_lock:
            self._model = AutoModelForObjectDetection.from_pretrained(
                artifact_path, config=model_config, device_map=self._device
            )
            self._model.eval()
        
        # Detect model type and set label offset
        self._model_name = type(self._model).__name__
        
        # Use config override if specified, otherwise auto-detect
        if self.config.requires_label_offset is not None:
            self._label_offset = 1 if self.config.requires_label_offset else 0
        else:
            # Auto-detect: RTDetr uses shifted labels with background at index 0
            if self._model_name == "RTDetrForObjectDetection":
                self._label_offset = 1
            else:
                self._label_offset = 0
        
        _log.debug(f"Loaded {self._model_name} with label_offset={self._label_offset}")
    
    def preprocess(
        self, images: List[Image.Image]
    ) -> Tuple[Any, torch.Tensor]:
        """Preprocess images using the image processor.
        
        Parameters
        ----------
        images : List[Image.Image]
            List of PIL images in RGB format
        
        Returns
        -------
        Tuple[Any, torch.Tensor]
            Preprocessed inputs and target sizes tensor
        """
        # Get target sizes (height, width) for each image
        target_sizes = torch.tensor([img.size[::-1] for img in images])
        
        # Preprocess images
        inputs = self._image_processor(images=images, return_tensors="pt").to(
            self._device
        )
        
        return inputs, target_sizes
    
    @torch.inference_mode()
    def infer(self, inputs: Any) -> Any:
        """Run inference on preprocessed inputs.
        
        Parameters
        ----------
        inputs : Any
            Preprocessed inputs from preprocess()
        
        Returns
        -------
        Any
            Raw model outputs
        """
        return self._model(**inputs)
    
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
            List of detection dicts with scores, labels, boxes
        """
        results: List[Dict[str, torch.Tensor]] = (
            self._image_processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=self.config.threshold,
            )
        )
        return results
    
    def get_label_offset(self) -> int:
        """Get label offset for this model.
        
        Returns
        -------
        int
            Label offset (1 for RTDetr, 0 for others)
        """
        return self._label_offset
    
    def info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns
        -------
        Dict[str, Any]
            Model configuration details
        """
        return {
            "model_type": "docling",
            "model_name": self._model_name,
            "device": self._device.type,
            "num_threads": self.config.num_threads,
            "image_size": self._image_processor.size if self._image_processor else None,
            "threshold": self.config.threshold,
            "label_offset": self._label_offset,
        }

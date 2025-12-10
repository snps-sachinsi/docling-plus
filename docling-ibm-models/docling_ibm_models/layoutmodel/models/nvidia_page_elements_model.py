#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
"""NVIDIA Nemotron Page Elements v3 layout detection model implementation."""

import logging
import os
import sys
import threading
from typing import Any, Dict, List, Tuple

import torch
import numpy as np
from PIL import Image

from docling_ibm_models.layoutmodel.layoutconfig import ModelConfig
from docling_ibm_models.layoutmodel.models.base import BaseLayoutModel

_log = logging.getLogger(__name__)

# Global lock for model initialization
_model_init_lock = threading.Lock()


class NvidiaPageElementsModel(BaseLayoutModel):
    """NVIDIA Nemotron Page Elements v3 object detection model.
    
    This model is a YOLOX-based detector specifically trained for identifying
    document elements: tables, charts, infographics, titles, headers/footers, and text.
    
    Output classes:
    - Table: Data structured in rows and columns
    - Chart: Bar charts, line charts, pie charts
    - Infographic: Complex visual representations (diagrams, flowcharts)
    - Title: Section titles or table/chart/infographic titles
    - Header/footer: Page headers and footers
    - Text: Regions of text paragraphs
    
    Example usage:
        predictor = LayoutPredictor(
            artifact_path="nvidia/nemotron-page-elements-v3",
            model_type="nvidia_page_elements",
        )
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize NVIDIA Page Elements model.
        
        Parameters
        ----------
        config : ModelConfig
            Model configuration
        """
        super().__init__(config)
        self._model = None
        self._model_name = "nvidia_page_elements_v3"
        self._label_offset = 0  # NVIDIA model uses 0-indexed labels
        self._model_dir = None
        
        # Class mapping: model internal name -> Docling label ID
        self._class_mapping = {
            "table": 4,      # Table
            "chart": 5,      # Picture/Chart
            "infographic": 5, # Picture/Infographic
            "title": 3,      # Title
            "header_footer": 9,  # Page-header (we'll use 9 for headers, could split)
            "text": 1,       # Text
        }
        
        # Load model during initialization
        self.load_model()
    
    def load_model(self) -> None:
        """Load NVIDIA Page Elements model from HuggingFace.
        
        Downloads the model if needed and initializes it.
        
        Raises
        ------
        FileNotFoundError
            If local model path is invalid
        ValueError
            If model loading fails
        """
        artifact_path = self.config.artifact_path
        
        # Download model if it's a HF repo ID
        if "/" in artifact_path and not os.path.exists(artifact_path):
            _log.info(f"Downloading NVIDIA Page Elements model: {artifact_path}")
            try:
                from huggingface_hub import snapshot_download
                self._model_dir = snapshot_download(repo_id=artifact_path)
                _log.info(f"Model downloaded to: {self._model_dir}")
            except Exception as e:
                raise ValueError(f"Failed to download model from {artifact_path}: {e}")
        else:
            if not os.path.exists(artifact_path):
                raise FileNotFoundError(f"Model path does not exist: {artifact_path}")
            self._model_dir = artifact_path
        
        # Add model directory to sys.path for custom code imports
        if self._model_dir not in sys.path:
            sys.path.append(self._model_dir)
        
        # Load model with thread safety
        with _model_init_lock:
            try:
                _log.info("Loading NVIDIA Page Elements model...")
                
                # Import the model definition from the cloned repo
                from nemotron_page_elements_v3.model import define_model
                
                self._model = define_model("page_element_v3")
                self._model = self._model.to(self._device).eval()
                
                _log.info("NVIDIA Page Elements model loaded successfully")
                
            except Exception as e:
                raise ValueError(f"Failed to load NVIDIA Page Elements model: {e}")
    
    def preprocess(
        self, images: List[Image.Image]
    ) -> Tuple[Any, torch.Tensor]:
        """Preprocess images for NVIDIA Page Elements model.
        
        Parameters
        ----------
        images : List[Image.Image]
            List of PIL images in RGB format
        
        Returns
        -------
        Tuple[Any, torch.Tensor]
            Preprocessed inputs (list of preprocessed arrays) and target sizes
        """
        # Get target sizes (height, width) for each image
        target_sizes = []
        preprocessed_inputs = []
        
        for img in images:
            # Ensure RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Convert to numpy array
            img_array = np.array(img)
            target_sizes.append(img_array.shape[:2])  # (height, width)
            
            # Preprocess using model's method
            preprocessed = self._model.preprocess(img_array)
            preprocessed_inputs.append(preprocessed)
        
        target_sizes = torch.tensor(target_sizes)
        
        return preprocessed_inputs, target_sizes
    
    @torch.inference_mode()
    def infer(self, inputs: Any) -> Any:
        """Run NVIDIA Page Elements model inference.
        
        Parameters
        ----------
        inputs : Any
            Preprocessed inputs (list of preprocessed arrays)
        
        Returns
        -------
        Any
            List of prediction arrays
        """
        _log.info("Starting NVIDIA Page Elements detection...")
        
        predictions = []
        for preprocessed_input in inputs:
            # Get original image shape from preprocessed data
            # The model expects (x, original_shape)
            # We'll pass a dummy shape and rely on postprocessing
            pred = self._model(preprocessed_input, (1024, 1024))[0]
            predictions.append(pred)
        
        _log.info(f"NVIDIA Page Elements detection completed ({len(predictions)} pages)")
        with open("./output/nvidia_page_elements_debug.log", "w") as debug_file:
            debug_file.write(str(predictions))
        return predictions
    
    def postprocess(
        self, outputs: Any, target_sizes: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """Post-process NVIDIA Page Elements outputs.
        
        Parameters
        ----------
        outputs : Any
            Raw model outputs (list of prediction arrays)
        target_sizes : torch.Tensor
            Target image sizes for rescaling boxes
        
        Returns
        -------
        List[Dict[str, torch.Tensor]]
            List of detection dicts with scores, labels, boxes
        """
        from nemotron_page_elements_v3.utils import postprocess_preds_page_element
        
        results = []
        for idx, pred in enumerate(outputs):
            height, width = target_sizes[idx].tolist()
            
            # Post-process predictions
            boxes, labels, scores = postprocess_preds_page_element(
                pred,
                self._model.thresholds_per_class,
                self._model.labels
            )
            
            if len(boxes) == 0:
                # No detections
                results.append({
                    "scores": torch.tensor([], dtype=torch.float32),
                    "labels": torch.tensor([], dtype=torch.int64),
                    "boxes": torch.tensor([], dtype=torch.float32).reshape(0, 4),
                })
                continue
            
            # Convert boxes from normalized to absolute coordinates
            # boxes are in format [x_min, y_min, x_max, y_max] normalized [0, 1]
            boxes_abs = boxes.copy()
            boxes_abs[:, [0, 2]] *= width   # x coordinates
            boxes_abs[:, [1, 3]] *= height  # y coordinates
            
            # Map class names to Docling label IDs
            label_ids = []
            for label_name in labels:
                label_id = self._class_mapping.get(label_name, 1)  # Default to Text
                label_ids.append(label_id)
            
            results.append({
                "scores": torch.tensor(scores, dtype=torch.float32),
                "labels": torch.tensor(label_ids, dtype=torch.int64),
                "boxes": torch.tensor(boxes_abs, dtype=torch.float32),
            })
            
            _log.info(f"Page {idx}: Detected {len(boxes)} elements")
        
        return results
    
    def get_label_offset(self) -> int:
        """Get label offset for NVIDIA Page Elements model.
        
        Returns
        -------
        int
            Label offset (0 for this model)
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
            "model_type": "nvidia_page_elements",
            "model_name": self._model_name,
            "model_dir": self._model_dir,
            "device": self._device.type,
            "num_threads": self.config.num_threads,
            "threshold": self.config.threshold,
            "label_offset": self._label_offset,
            "classes": list(self._class_mapping.keys()),
            "status": "implemented",
        }

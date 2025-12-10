#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import logging
from collections.abc import Iterable
from typing import Dict, List, Optional, Set, Union

import numpy as np
import torch
from PIL import Image

from docling_ibm_models.layoutmodel.layoutconfig import get_model_config
from docling_ibm_models.layoutmodel.models.factory import LayoutModelFactory
from docling_ibm_models.layoutmodel.models.postprocessing import (
    DetectionPostProcessor,
)
from docling_ibm_models.layoutmodel.labels import LayoutLabels

_log = logging.getLogger(__name__)


class LayoutPredictor:
    """
    Document layout prediction using safe tensors.
    
    Supports multiple layout detection models through factory pattern.
    Different models like Docling, NVIDIA NeMo, etc. can be used.
    """

    def __init__(
        self,
        artifact_path: str,
        device: str = "cpu",
        num_threads: int = 4,
        base_threshold: float = 0.3,
        blacklist_classes: Set[str] = set(),
        model_type: str = "docling",
        model_config: Optional[Dict[str, any]] = None,
    ):
        """
        Provide the artifact path that contains the LayoutModel file

        Parameters
        ----------
        artifact_path: Path for the model artifacts.
        device: (Optional) device to run the inference.
        num_threads: (Optional) Number of threads to run the inference if device = 'cpu'
        base_threshold: (Optional) confidence threshold for predictions
        blacklist_classes: (Optional) classes to filter out from predictions
        model_type: (Optional) layout model type to use (default: "docling")
                    Options: "docling", "nvidia_nemo", or custom registered types
        model_config: (Optional) configuration dict to override defaults for this model.
                     If provided, explicit params (device, num_threads, etc.) are ignored.
                     Config keys: device, num_threads, threshold, blacklist_classes, etc.

        Raises
        ------
        FileNotFoundError when the model's files are missing
        ValueError when model_type is not supported
        """
        # If model_config not provided, build from explicit parameters
        if model_config is None:
            model_config = {
                "device": device,
                "num_threads": num_threads,
                "threshold": base_threshold,
                "blacklist_classes": blacklist_classes,
            }
        
        # Get merged config (defaults + model-specific + user)
        merged_config = get_model_config(model_type=model_type, user_config=model_config)
        
        # Extract final values
        final_device = merged_config.get("device", "cpu")
        final_num_threads = merged_config.get("num_threads", 4)
        final_threshold = merged_config.get("threshold", 0.3)
        final_blacklist = merged_config.get("blacklist_classes", set())
        
        # Store parameters
        self._black_classes = final_blacklist
        self._threshold = final_threshold
        self._device = torch.device(final_device)
        self._model_config = merged_config
        
        # Canonical classes
        self._labels = LayoutLabels()
        
        # Create layout model using factory
        self._model = LayoutModelFactory.create_model(
            model_type=model_type,
            artifact_path=artifact_path,
            device=final_device,
            num_threads=final_num_threads,
            threshold=final_threshold,
            blacklist_classes=final_blacklist,
            model_config=merged_config,  # Pass full config to factory for underlying modules
        )
        
        # Get label offset and classes map from model
        label_offset = self._model.get_label_offset()
        if label_offset == 1:
            self._classes_map = self._labels.shifted_canonical_categories()
        else:
            self._classes_map = self._labels.canonical_categories()
        self._label_offset = label_offset
        
        _log.debug("LayoutPredictor settings: {}".format(self.info()))

    def info(self) -> dict:
        """
        Get information about the configuration of LayoutPredictor
        """
        model_info = self._model.info()
        info = {
            "model_type": model_info.get("model_type", "unknown"),
            "model_name": model_info.get("model_name", "unknown"),
            "device": self._device.type,
            "threshold": self._threshold,
            "label_offset": self._label_offset,
            "model_config": self._model_config,
        }
        info.update(model_info)
        return info

    @torch.inference_mode()
    def predict(self, orig_img: Union[Image.Image, np.ndarray]) -> Iterable[dict]:
        """
        Predict bounding boxes for a given image.
        The origin (0, 0) is the top-left corner and the predicted bbox coords are provided as:
        [left, top, right, bottom]

        Parameter
        ---------
        origin_img: Image to be predicted as a PIL Image object or numpy array.

        Yield
        -----
        Bounding box as a dict with the keys: "label", "confidence", "l", "t", "r", "b"

        Raises
        ------
        TypeError when the input image is not supported
        """
        # Use predict_batch and yield results
        results = self.predict_batch([orig_img])
        if results:
            for prediction in results[0]:
                yield prediction

    @torch.inference_mode()
    def predict_batch(
        self, images: List[Union[Image.Image, np.ndarray]]
    ) -> List[List[dict]]:
        """
        Batch prediction for multiple images - more efficient than calling predict() multiple times.

        Parameters
        ----------
        images : List[Union[Image.Image, np.ndarray]]
            List of images to process in a single batch

        Returns
        -------
        List[List[dict]]
            List of prediction lists, one per input image. Each prediction dict contains:
            "label", "confidence", "l", "t", "r", "b"
        """
        if not images:
            return []

        # Convert all images to RGB PIL format
        pil_images = []
        for img in images:
            if isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            elif isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img).convert("RGB"))
            else:
                raise TypeError("Not supported input image format")

        # Use layout model for preprocessing, inference, and postprocessing
        inputs, target_sizes = self._model.preprocess(pil_images)
        outputs = self._model.infer(inputs)
        results_list = self._model.postprocess(outputs, target_sizes)

        # Format detections to standard output format
        all_predictions = DetectionPostProcessor.format_detections(
            results_list=results_list,
            images=pil_images,
            classes_map=self._classes_map,
            label_offset=self._label_offset,
            blacklist_classes=self._black_classes,
        )

        return all_predictions

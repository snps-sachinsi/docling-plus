#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
"""NVIDIA NeMo layout detection model implementation."""

import logging
import os
import sys
import threading
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, GenerationConfig
from huggingface_hub import snapshot_download

from docling_ibm_models.layoutmodel.layoutconfig import ModelConfig
from docling_ibm_models.layoutmodel.models.base import BaseLayoutModel

_log = logging.getLogger(__name__)

# Global lock for model initialization to prevent threading issues
_model_init_lock = threading.Lock()


class NvidiaLayoutModel(BaseLayoutModel):
    """NVIDIA NeMo Parse layout detection model implementation.
    
    This implementation integrates NVIDIA Nemotron Parse models for document
    layout analysis. It supports the NVIDIA-Nemotron-Parse-v1.1 model which
    generates markdown output with bounding boxes and class labels.
    
    Example usage:
        # Register the model
        LayoutModelFactory.register_model("nvidia_nemo", NvidiaLayoutModel)
        
        # Use it
        predictor = LayoutPredictor(
            artifact_path="nvidia/NVIDIA-Nemotron-Parse-v1.1",  # HF repo ID
            model_type="nvidia_nemo",
        )
    
    Note:
        The artifact_path should be a HuggingFace repository ID like
        "nvidia/NVIDIA-Nemotron-Parse-v1.1" or a local path to a downloaded model.
    """
    
    # NVIDIA prompts for different use cases
    # Standard prompt with bounding boxes and classes
    NVIDIA_PROMPT_STANDARD = "</s><s><predict_bbox><predict_classes><output_markdown>"
    
    # Alternative prompt that may better detect tables/figures
    # This is the default for comprehensive document understanding
    NVIDIA_PROMPT = NVIDIA_PROMPT_STANDARD
    
    def __init__(self, config: ModelConfig):
        """Initialize NVIDIA layout model.
        
        Parameters
        ----------
        config : ModelConfig
            Model configuration
        """
        super().__init__(config)
        self._model = None
        self._processor = None
        self._generation_config = None
        self._model_name = "nvidia_nemo_parse"
        self._label_offset = 0  # NVIDIA model uses 0-indexed labels
        self._model_dir = None
        
        # Load model during initialization
        self.load_model()
    
    def load_model(self) -> None:
        """Load NVIDIA NeMo model from HuggingFace or local path.
        
        Downloads the model snapshot if needed and loads the model with processor.
        Requires trust_remote_code=True as NVIDIA models use custom code.
        
        Raises
        ------
        FileNotFoundError
            If local model path is invalid
        ValueError
            If model loading fails
        """
        artifact_path = self.config.artifact_path
        
        # Download model snapshot if artifact_path looks like a HF repo ID
        if "/" in artifact_path and not os.path.exists(artifact_path):
            _log.info(f"Downloading NVIDIA model from HuggingFace: {artifact_path}")
            try:
                self._model_dir = snapshot_download(repo_id=artifact_path)
                _log.info(f"Model downloaded to: {self._model_dir}")
            except Exception as e:
                raise ValueError(f"Failed to download model from {artifact_path}: {e}")
        else:
            # Use local path
            if not os.path.exists(artifact_path):
                raise FileNotFoundError(f"Model path does not exist: {artifact_path}")
            self._model_dir = artifact_path
        
        # Add model directory to sys.path for custom code imports
        if self._model_dir not in sys.path:
            sys.path.append(self._model_dir)
        
        # Determine dtype based on device
        dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        
        # Load model and processor with thread safety
        with _model_init_lock:
            try:
                _log.info("Loading NVIDIA model and processor...")
                self._model = AutoModel.from_pretrained(
                    self._model_dir,
                    trust_remote_code=True,
                    torch_dtype=dtype
                ).to(self._device).eval()
                
                self._processor = AutoProcessor.from_pretrained(
                    self._model_dir,
                    trust_remote_code=True
                )
                
                # Load generation config
                try:
                    self._generation_config = GenerationConfig.from_pretrained(
                        self._model_dir,
                        trust_remote_code=True
                    )
                    self._generation_config.use_cache = False
                    # Set reasonable max_new_tokens to avoid excessive generation time
                    if not hasattr(self._generation_config, 'max_new_tokens') or self._generation_config.max_new_tokens > 2048:
                        self._generation_config.max_new_tokens = 2048
                    _log.debug("Generation config loaded successfully")
                except Exception as e:
                    _log.warning(f"Could not load GenerationConfig: {e}. Using default.")
                    self._generation_config = GenerationConfig(max_new_tokens=2048, use_cache=False)
                
                _log.info("NVIDIA model loaded successfully")
                
            except Exception as e:
                raise ValueError(f"Failed to load NVIDIA model: {e}")
    
    def preprocess(
        self, images: List[Image.Image]
    ) -> Tuple[Any, torch.Tensor]:
        """Preprocess images for NVIDIA model.
        
        Uses NVIDIA's processor to prepare images with the official prompt.
        
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
        
        # Ensure images are RGB
        images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
        
        # Process with NVIDIA prompt
        inputs = self._processor(
            images=images,
            text=self.NVIDIA_PROMPT,
            return_tensors="pt"
        ).to(self._device)
        
        # Convert float32 to bfloat16/float16 if on CUDA for consistency
        if self._device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            inputs = {
                k: v.to(dtype) if v.dtype == torch.float32 else v
                for k, v in inputs.items()
            }
        
        return inputs, target_sizes
    
    @torch.inference_mode()
    def infer(self, inputs: Any) -> Any:
        """Run NVIDIA model inference.
        
        Generates structured output with bounding boxes and class labels.
        
        Parameters
        ----------
        inputs : Any
            Preprocessed inputs from preprocess()
        
        Returns
        -------
        Any
            Generated output sequences (token IDs)
        """
        _log.info("Starting NVIDIA model generation (this may take a moment)...")
        # Use optimized generation config with caching enabled
        gen_kwargs = {
            "generation_config": self._generation_config,
            "do_sample": False,  # Use greedy decoding for speed
            "num_beams": 1,  # No beam search for faster generation
        }
        output_sequences = self._model.generate(**inputs, **gen_kwargs)
        # Save output sequences in human readable format
        os.makedirs("./output", exist_ok=True)
        for i, seq in enumerate(output_sequences):
            decoded = self._processor.decode(seq, skip_special_tokens=True)
            with open(f"./output/infer_output_{i}.txt", "w", encoding="utf-8") as f:
                f.write(decoded)
        _log.info("NVIDIA model generation completed")
        return output_sequences
    
    def postprocess(
        self, outputs: Any, target_sizes: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """Post-process NVIDIA model outputs.
        
        Decodes the generated sequences and parses the structured output
        to extract bounding boxes, labels, and scores.
        
        Parameters
        ----------
        outputs : Any
            Raw model outputs from infer() (token sequences)
        target_sizes : torch.Tensor
            Target sizes for rescaling boxes
        
        Returns
        -------
        List[Dict[str, torch.Tensor]]
            List of detection dicts with scores, labels, boxes
            
        Note
        ----
        NVIDIA Nemotron Parse generates text output in a structured format
        with bounding boxes and class labels. This method parses that output
        and converts it to the standard detection dictionary format.
        """
        # Decode the output sequences
        decoded_results = self._processor.batch_decode(outputs, skip_special_tokens=True)
        
        # Parse the structured output to extract detections
        # NVIDIA output format includes bounding boxes and class labels in text
        results = []
        for idx, result_text in enumerate(decoded_results):
            detections = self._parse_nvidia_output(result_text, target_sizes[idx])
            results.append(detections)
        
        return results
    
    def _parse_nvidia_output(
        self, result_text: str, target_size: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Parse NVIDIA structured output to extract detections.
        
        Parameters
        ----------
        result_text : str
            Decoded model output text
        target_size : torch.Tensor
            Target image size [height, width]
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Detection dict with scores, labels, boxes
            
        Note
        ----
        NVIDIA Nemotron Parse outputs structured format:
        <x_0.3438><y_0.3312>text content<x_0.4961><y_0.3891><class_Text>
        Where x/y are normalized coordinates (0-1) and class is the element type.
        """
        import re
        
        # Define class name to label ID mapping (matching Docling's LayoutLabels)
        class_to_label = {
            "Text": 1,
            "Section-header": 2,
            "Title": 3,
            "Table": 4,
            "Picture": 5,
            "Caption": 6,
            "Formula": 7,
            "Footnote": 8,
            "Page-header": 9,
            "Page-footer": 10,
            "List-item": 11,
            "Code": 12,
            "Figure": 5,  # Map to Picture
            "Image": 5,   # Map to Picture
        }
        
        # Parse structured output - pattern: <x_start><y_start>...content...<x_end><y_end><class_Name>
        # Looking for: <x_float><y_float>content<x_float><y_float><class_Name>
        pattern = r'<x_([\d.]+)><y_([\d.]+)>.*?<x_([\d.]+)><y_([\d.]+)><class_([^>]+)>'
        matches = re.findall(pattern, result_text)
        
        if not matches:
            _log.debug("No structured bounding boxes found in NVIDIA output")
            return {
                "scores": torch.tensor([], dtype=torch.float32),
                "labels": torch.tensor([], dtype=torch.int64),
                "boxes": torch.tensor([], dtype=torch.float32).reshape(0, 4),
            }
        
        boxes = []
        labels = []
        scores = []
        
        height, width = target_size[0].item(), target_size[1].item()
        
        for match in matches:
            x1_norm, y1_norm, x2_norm, y2_norm, class_name = match
            
            try:
                # Convert normalized coords to absolute pixel coords
                x1 = float(x1_norm) * width
                y1 = float(y1_norm) * height
                x2 = float(x2_norm) * width
                y2 = float(y2_norm) * height
                
                # Get label ID (default to 1 for Text if unknown)
                label_id = class_to_label.get(class_name, 1)
                
                # Create box in [x1, y1, x2, y2] format
                boxes.append([x1, y1, x2, y2])
                labels.append(label_id)
                # NVIDIA doesn't provide confidence scores, use 1.0
                scores.append(1.0)
                
            except (ValueError, IndexError) as e:
                _log.debug(f"Failed to parse detection: {e}")
                continue
        
        if not boxes:
            return {
                "scores": torch.tensor([], dtype=torch.float32),
                "labels": torch.tensor([], dtype=torch.int64),
                "boxes": torch.tensor([], dtype=torch.float32).reshape(0, 4),
            }
        
        _log.info(f"Parsed {len(boxes)} detections from NVIDIA output")
        _log.debug(f"Detected classes: {set(class_to_label.get(m[4], 'Unknown') for m in matches)}")
        
        return {
            "scores": torch.tensor(scores, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
        }
    
    def get_label_offset(self) -> int:
        """Get label offset for NVIDIA model.
        
        Returns
        -------
        int
            Label offset (typically 0 or 1)
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
            "model_type": "nvidia_nemo",
            "model_name": self._model_name,
            "model_dir": self._model_dir,
            "device": self._device.type,
            "num_threads": self.config.num_threads,
            "threshold": self.config.threshold,
            "label_offset": self._label_offset,
            "prompt": self.NVIDIA_PROMPT,
            "status": "implemented",
        }

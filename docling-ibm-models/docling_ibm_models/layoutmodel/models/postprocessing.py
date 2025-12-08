#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
"""Post-processing utilities for detection outputs."""

from typing import Dict, List, Set

import torch
from PIL import Image


class DetectionPostProcessor:
    """Utilities for post-processing detection model outputs."""
    
    @staticmethod
    def format_detections(
        results_list: List[Dict[str, torch.Tensor]],
        images: List[Image.Image],
        classes_map: Dict[int, str],
        label_offset: int = 0,
        blacklist_classes: Set[str] = None,
    ) -> List[List[dict]]:
        """Convert raw detection tensors to standardized format.
        
        Parameters
        ----------
        results_list : List[Dict[str, torch.Tensor]]
            List of detection results, each dict containing:
            - "scores": Tensor of confidence scores
            - "labels": Tensor of class label IDs
            - "boxes": Tensor of bounding boxes [x1, y1, x2, y2]
        images : List[Image.Image]
            Original PIL images (for size reference)
        classes_map : Dict[int, str]
            Mapping from label IDs to class names
        label_offset : int, optional
            Offset to add to label IDs before mapping, by default 0
        blacklist_classes : Set[str], optional
            Classes to filter out, by default None
        
        Returns
        -------
        List[List[dict]]
            List of predictions per image, each prediction dict contains:
            - "l": left coordinate
            - "t": top coordinate
            - "r": right coordinate
            - "b": bottom coordinate
            - "label": class label string
            - "confidence": confidence score
        """
        if blacklist_classes is None:
            blacklist_classes = set()
        
        all_predictions = []
        
        for img, results in zip(images, results_list):
            w, h = img.size
            predictions = []
            
            for score, label_id, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                score = float(score.item())
                label_id = int(label_id.item()) + label_offset
                label_str = classes_map[label_id]
                
                # Filter out blacklisted classes
                if label_str in blacklist_classes:
                    continue
                
                # Extract and clamp bounding box coordinates
                bbox_float = [float(b.item()) for b in box]
                l = min(w, max(0, bbox_float[0]))
                t = min(h, max(0, bbox_float[1]))
                r = min(w, max(0, bbox_float[2]))
                b = min(h, max(0, bbox_float[3]))
                
                predictions.append(
                    {
                        "l": l,
                        "t": t,
                        "r": r,
                        "b": b,
                        "label": label_str,
                        "confidence": score,
                    }
                )
            
            all_predictions.append(predictions)
        
        return all_predictions

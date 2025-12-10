"""Example: Extending SynDoc with custom models."""

import logging
from typing import Any, Dict, List

import torch
from PIL import Image

from syndoc import DocumentConverter, ConversionConfig, ModelConfig, PipelineConfig
from syndoc.models.base import BaseLayoutModel
from syndoc.models.factory import ModelRegistry

logging.basicConfig(level=logging.INFO)


class CustomLayoutModel(BaseLayoutModel):
    """Custom layout detection model.
    
    This is an example showing how to integrate your own model
    (e.g., a custom YOLO, DETR, or other detection model).
    """
    
    def load_model(self) -> None:
        """Load your custom model.
        
        This could load from HuggingFace, a local checkpoint, etc.
        """
        print(f"Loading custom model from: {self.config.artifact_path}")
        
        # Example: Load a pretrained model
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        # self.model.to(self.device)
        # self.model.eval()
        
        # For this example, we'll just use a placeholder
        self.model = None
        print("Custom model loaded successfully")
    
    def predict(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Run prediction on images.
        
        Args:
            images: List of PIL images
            
        Returns:
            List of predictions with boxes, labels, scores
        """
        predictions = []
        
        for image in images:
            # Example: Run your model
            # with torch.no_grad():
            #     results = self.model(image)
            #     boxes = results.xyxy[0][:, :4].cpu().numpy()
            #     scores = results.xyxy[0][:, 4].cpu().numpy()
            #     labels = results.xyxy[0][:, 5].cpu().numpy()
            
            # For this example, return dummy predictions
            width, height = image.size
            predictions.append({
                "boxes": [
                    [0.1 * width, 0.1 * height, 0.9 * width, 0.3 * height],
                    [0.1 * width, 0.4 * height, 0.9 * width, 0.6 * height],
                ],
                "labels": ["title", "text"],
                "scores": [0.95, 0.87],
            })
        
        return predictions


def register_custom_model():
    """Register custom model with SynDoc."""
    print("\n=== Registering Custom Model ===\n")
    
    # Register the model
    ModelRegistry.register_layout_model("custom", CustomLayoutModel)
    print("Custom model registered successfully")


def use_custom_model():
    """Use custom model in conversion."""
    print("\n=== Using Custom Model ===\n")
    
    # Create configuration with custom model
    config = ConversionConfig(
        pipeline=PipelineConfig(
            do_layout_detection=True,
            layout_model=ModelConfig(
                model_type="custom",  # Use our custom model
                artifact_path="/path/to/model/weights",
                device="cpu",
                threshold=0.5,
                custom_params={
                    "dpi": 144,
                    "model_version": "v1.0",
                },
            ),
        ),
    )
    
    converter = DocumentConverter(config=config)
    
    # Convert document
    result = converter.convert("sample.pdf")
    
    print(f"Status: {result.status.value}")
    print(f"Pages: {len(result.pages)}")
    
    if result.pages:
        print(f"\nElements from first page:")
        for elem in result.pages[0].elements:
            print(f"  {elem.element_type.value}: confidence={elem.confidence:.2f}")


class CustomDetectionModel(BaseLayoutModel):
    """Example: Integration with a real detection model.
    
    This shows the pattern for integrating models like:
    - DETR (Detection Transformer)
    - Faster R-CNN
    - YOLO variants
    - Custom trained models
    """
    
    def load_model(self) -> None:
        """Load detection model.
        
        Example with transformers DETR:
        
        from transformers import DetrImageProcessor, DetrForObjectDetection
        
        self.processor = DetrImageProcessor.from_pretrained(
            self.config.artifact_path
        )
        self.model = DetrForObjectDetection.from_pretrained(
            self.config.artifact_path
        )
        self.model.to(self.device)
        self.model.eval()
        """
        print("Loading detection model...")
        # Implementation here
        self.model = None
        self.processor = None
    
    def predict(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Run detection.
        
        Example with transformers DETR:
        
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process outputs
        target_sizes = torch.tensor([img.size[::-1] for img in images])
        results = self.processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes,
            threshold=self.config.threshold
        )
        
        predictions = []
        for result in results:
            predictions.append({
                "boxes": result["boxes"].cpu().numpy().tolist(),
                "labels": result["labels"].cpu().numpy().tolist(),
                "scores": result["scores"].cpu().numpy().tolist(),
            })
        
        return predictions
        """
        # Placeholder implementation
        return [{"boxes": [], "labels": [], "scores": []} for _ in images]


def example_model_integration_patterns():
    """Show various model integration patterns."""
    print("\n=== Model Integration Patterns ===\n")
    
    # Pattern 1: HuggingFace model
    print("1. HuggingFace Integration:")
    print("   model_type: 'detr'")
    print("   artifact_path: 'facebook/detr-resnet-50'")
    
    # Pattern 2: Local checkpoint
    print("\n2. Local Checkpoint:")
    print("   model_type: 'custom'")
    print("   artifact_path: '/path/to/checkpoint.pth'")
    
    # Pattern 3: ONNX model
    print("\n3. ONNX Model:")
    print("   model_type: 'onnx'")
    print("   artifact_path: '/path/to/model.onnx'")
    print("   custom_params: {'use_onnx': True}")
    
    # Pattern 4: Multi-model pipeline
    print("\n4. Multi-Model Pipeline:")
    config = ConversionConfig(
        pipeline=PipelineConfig(
            do_layout_detection=True,
            do_ocr=True,
            do_table_structure=True,
            layout_model=ModelConfig(
                model_type="custom",
                device="cuda",
            ),
            ocr_model=ModelConfig(
                model_type="mock",
                device="cpu",
            ),
            table_model=ModelConfig(
                model_type="mock",
                device="cpu",
            ),
        ),
    )
    print(f"   Pipeline with {3} models configured")


if __name__ == "__main__":
    print("SynDoc Custom Model Examples")
    print("=" * 50)
    
    # Register and use custom model
    register_custom_model()
    # use_custom_model()  # Uncomment with real PDF
    
    # Show integration patterns
    example_model_integration_patterns()
    
    print("\n" + "=" * 50)
    print("See the CustomLayoutModel class above for implementation details")

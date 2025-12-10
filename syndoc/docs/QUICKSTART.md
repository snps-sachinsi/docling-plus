# Quick Start Guide

## Installation

```bash
cd syndoc
pip install -e .
```

## Basic Usage

### Simple Conversion

```python
from syndoc import DocumentConverter

# Create converter
converter = DocumentConverter()

# Convert a PDF
result = converter.convert("document.pdf")

# Check results
print(f"Status: {result.status}")
print(f"Pages: {len(result.pages)}")
print(f"Errors: {len(result.errors)}")
```

### With Configuration

```python
from syndoc import DocumentConverter, ConversionConfig, PipelineConfig

config = ConversionConfig(
    pipeline=PipelineConfig(
        do_layout_detection=True,
        do_ocr=False,
    ),
    enable_profiling=True,
)

converter = DocumentConverter(config=config)
result = converter.convert("document.pdf")

# Access profiling info
if result.profiling:
    print(f"Time: {result.profiling.total_time:.2f}s")
    print(f"Speed: {result.profiling.pages_per_second:.2f} pages/s")
```

### Accessing Page Elements

```python
result = converter.convert("document.pdf")

for page in result.pages:
    print(f"\nPage {page.page_no}:")
    print(f"  Size: {page.size.width} x {page.size.height}")
    
    for element in page.elements:
        print(f"  - {element.element_type.value}")
        print(f"    BBox: ({element.bbox.x0:.2f}, {element.bbox.y0:.2f}) "
              f"to ({element.bbox.x1:.2f}, {element.bbox.y1:.2f})")
        if element.text:
            print(f"    Text: {element.text[:50]}...")
```

### Export Results

```python
result = converter.convert("document.pdf")

# To dictionary
data = result.to_dict()

# To JSON
json_str = result.to_json()
with open("result.json", "w") as f:
    f.write(json_str)
```

## Advanced Usage

### Custom Model Integration

```python
from syndoc.models.base import BaseLayoutModel
from syndoc.models.factory import ModelRegistry
from syndoc import ModelConfig

# Define custom model
class MyModel(BaseLayoutModel):
    def load_model(self):
        # Load your model
        pass
    
    def predict(self, images):
        # Run inference
        return predictions

# Register model
ModelRegistry.register_layout_model("my_model", MyModel)

# Use in config
config = ConversionConfig(
    pipeline=PipelineConfig(
        do_layout_detection=True,
        layout_model=ModelConfig(
            model_type="my_model",
            artifact_path="/path/to/weights",
            device="cuda",
            threshold=0.5,
        ),
    ),
)

converter = DocumentConverter(config=config)
```

### Batch Processing

```python
import glob
from pathlib import Path

converter = DocumentConverter()

pdf_files = glob.glob("documents/*.pdf")
results = []

for pdf_file in pdf_files:
    print(f"Processing {pdf_file}...")
    result = converter.convert(pdf_file)
    results.append(result)
    
    if result.status == "success":
        print(f"  ✓ Success: {len(result.pages)} pages")
    else:
        print(f"  ✗ Failed: {len(result.errors)} errors")

print(f"\nTotal: {len(results)} documents")
print(f"Success: {sum(1 for r in results if r.status == 'success')}")
```

### Error Handling

```python
from syndoc import ConversionConfig, ConversionStatus

# Configure to capture errors (not raise)
config = ConversionConfig(raises_on_error=False)
converter = DocumentConverter(config=config)

result = converter.convert("document.pdf")

if result.status == ConversionStatus.FAILURE:
    print("Conversion failed!")
    for error in result.errors:
        print(f"  {error.stage}: {error.message}")
        if error.page_no:
            print(f"    (on page {error.page_no})")
elif result.status == ConversionStatus.PARTIAL_SUCCESS:
    print(f"Partial success: {result.success_pages} pages processed")
    print(f"Errors: {result.error_count}")
else:
    print(f"Success: {result.success_pages} pages")
```

## Configuration Options

### ConversionConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | `PipelineConfig` | Default | Pipeline configuration |
| `max_file_size` | `int` | 100000000 | Max file size in bytes |
| `max_num_pages` | `int` | 10000 | Max pages to process |
| `page_range` | `tuple` | (1, None) | Page range (1-indexed) |
| `raises_on_error` | `bool` | False | Raise exceptions |
| `enable_profiling` | `bool` | False | Track performance |

### PipelineConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `do_layout_detection` | `bool` | True | Enable layout detection |
| `do_ocr` | `bool` | False | Enable OCR |
| `do_table_structure` | `bool` | False | Enable table parsing |
| `do_reading_order` | `bool` | False | Enable reading order |
| `layout_model` | `ModelConfig` | None | Layout model config |
| `ocr_model` | `ModelConfig` | None | OCR model config |
| `table_model` | `ModelConfig` | None | Table model config |
| `artifacts_path` | `Path` | None | Base path for models |

### ModelConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | `str` | Required | Model identifier |
| `artifact_path` | `str` | None | Path to model files |
| `device` | `DeviceType` | CPU | Compute device |
| `threshold` | `float` | 0.5 | Confidence threshold |
| `batch_size` | `int` | 1 | Inference batch size |
| `num_threads` | `int` | 4 | CPU threads |
| `blacklist_classes` | `set` | {} | Classes to ignore |
| `custom_params` | `dict` | {} | Model-specific params |

## Common Patterns

### Configuration from Dict

```python
config_dict = {
    "pipeline": {
        "do_layout_detection": True,
        "layout_model": {
            "model_type": "detr",
            "device": "cuda",
            "threshold": 0.6,
        }
    },
    "max_file_size": 50000000,
    "enable_profiling": True,
}

converter = DocumentConverter.from_config_dict(config_dict)
```

### Processing Specific Pages

```python
config = ConversionConfig(
    page_range=(1, 10),  # Only first 10 pages
)

converter = DocumentConverter(config=config)
result = converter.convert("large_document.pdf")
```

### Filtering Elements

```python
from syndoc import ElementType

result = converter.convert("document.pdf")

for page in result.pages:
    # Get only tables
    tables = [e for e in page.elements 
              if e.element_type == ElementType.TABLE]
    
    # Get only high-confidence elements
    high_conf = [e for e in page.elements 
                 if e.confidence > 0.9]
    
    # Get text elements
    texts = [e for e in page.elements 
             if e.element_type == ElementType.TEXT]
```

## Troubleshooting

### "Module not found"
```bash
pip install -e .
```

### "PDF not found"
```python
from pathlib import Path

pdf_path = Path("document.pdf")
if not pdf_path.exists():
    print(f"File not found: {pdf_path.absolute()}")
```

### "CUDA out of memory"
```python
config = ConversionConfig(
    pipeline=PipelineConfig(
        layout_model=ModelConfig(
            device="cpu",  # Use CPU instead
            batch_size=1,
        )
    )
)
```

### "Invalid PDF"
```python
result = converter.convert("document.pdf")

if not result.input.valid:
    print("Invalid PDF file")
    print(f"Errors: {result.errors}")
```

## Next Steps

- Read [ARCHITECTURE.md](docs/ARCHITECTURE.md) for design details
- See [examples/](examples/) for more examples
- Check [README.md](README.md) for full documentation

## Support

For issues and questions:
1. Check existing documentation
2. Review example code
3. Examine error messages in `result.errors`

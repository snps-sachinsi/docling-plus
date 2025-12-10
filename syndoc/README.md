# SynDoc - Synopsys Document Parser

**SynDoc** is a scalable and extensible document parsing library designed for converting PDF documents into structured representations. Inspired by the docling architecture, SynDoc provides a flexible pipeline-based approach for document understanding.

## Features

- **Modular Architecture**: Clean separation between backends, pipelines, and models
- **Extensible Design**: Easy to add new document formats, models, and processing stages
- **Configuration-Driven**: Flexible configuration system for customizing behavior
- **Type-Safe**: Fully typed with Pydantic models for validation and clarity
- **Pipeline-Based**: Multi-stage processing for layout detection, text extraction, and enrichment

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   DocumentConverter                     │
│  (Orchestrates conversion process)                      │
└─────────────────────┬───────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
┌────────▼─────────┐      ┌───────▼──────────┐
│  Backend Layer   │      │  Pipeline Layer  │
│  (Document I/O)  │      │  (Processing)    │
└────────┬─────────┘      └───────┬──────────┘
         │                         │
    ┌────▼────┐            ┌──────▼───────┐
    │  PDF    │            │ Layout Model │
    │ Backend │            │ OCR Model    │
    └─────────┘            │ Table Model  │
                           └──────────────┘
```

### Components

1. **Data Models** (`syndoc/datamodels/`):
   - Document representations (InputDocument, Page, ConversionResult)
   - Configuration options
   - Status and error handling

2. **Backends** (`syndoc/backends/`):
   - Abstract base classes for document I/O
   - Concrete implementations (PDF, etc.)
   - Page-level access and metadata extraction

3. **Pipelines** (`syndoc/pipelines/`):
   - Abstract pipeline interface
   - PDF processing pipeline with configurable stages
   - Layout detection, OCR, table extraction

4. **Models** (`syndoc/models/`):
   - Abstract model interfaces
   - Configuration system for different model types
   - Support for custom model implementations

## Installation

```bash
# Basic installation
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from syndoc import DocumentConverter
from syndoc.config import ConversionConfig, PipelineConfig

# Create converter with default configuration
converter = DocumentConverter()

# Convert a PDF document
result = converter.convert("document.pdf")

# Access the parsed content
print(f"Status: {result.status}")
print(f"Pages: {len(result.pages)}")
for page in result.pages:
    print(f"Page {page.page_no}: {len(page.elements)} elements")

# Custom configuration
config = ConversionConfig(
    pipeline=PipelineConfig(
        do_layout_detection=True,
        do_ocr=False,
        layout_model_type="detr",
    )
)
converter = DocumentConverter(config=config)
result = converter.convert("document.pdf")
```

## Configuration

SynDoc provides flexible configuration at multiple levels:

```python
from syndoc.config import ConversionConfig, PipelineConfig, ModelConfig

config = ConversionConfig(
    # Pipeline configuration
    pipeline=PipelineConfig(
        do_layout_detection=True,
        do_ocr=True,
        do_table_structure=True,
        
        # Model configurations
        layout_model=ModelConfig(
            model_type="detr",
            device="cuda",
            threshold=0.5,
        ),
        
        ocr_model=ModelConfig(
            model_type="tesseract",
            device="cpu",
        ),
    ),
    
    # Processing options
    max_file_size=100_000_000,  # 100 MB
    page_range=(1, None),  # All pages
)
```

## Extension Points

### Adding a New Backend

```python
from syndoc.backends.base import AbstractDocumentBackend

class MyBackend(AbstractDocumentBackend):
    def is_valid(self) -> bool:
        # Validate document
        pass
    
    def load_page(self, page_no: int) -> Page:
        # Load and return page
        pass
```

### Adding a New Model

```python
from syndoc.models.base import BaseModel

class MyLayoutModel(BaseModel):
    def load_model(self) -> None:
        # Initialize model
        pass
    
    def predict(self, page_image):
        # Run inference
        pass
```

### Custom Pipeline Stage

```python
from syndoc.pipelines.base import BasePipeline

class MyPipeline(BasePipeline):
    def _process_page(self, page: Page) -> Page:
        # Custom processing
        return page
```

## Project Structure

```
syndoc/
├── __init__.py              # Public API
├── converter.py             # DocumentConverter
├── backends/                # Document backends
│   ├── __init__.py
│   ├── base.py             # Abstract base
│   └── pdf_backend.py      # PDF implementation
├── datamodels/             # Data structures
│   ├── __init__.py
│   ├── document.py         # Document models
│   ├── config.py           # Configuration
│   └── enums.py            # Enumerations
├── pipelines/              # Processing pipelines
│   ├── __init__.py
│   ├── base.py             # Base pipeline
│   └── pdf_pipeline.py     # PDF pipeline
├── models/                 # Model implementations
│   ├── __init__.py
│   ├── base.py             # Base model interface
│   └── config.py           # Model configuration
└── utils/                  # Utilities
    ├── __init__.py
    └── profiling.py        # Performance tracking
```

## Design Principles

1. **Separation of Concerns**: Clear boundaries between I/O (backends), processing (pipelines), and ML (models)
2. **Open/Closed Principle**: Open for extension through abstract base classes, closed for modification
3. **Dependency Inversion**: Depend on abstractions, not concrete implementations
4. **Configuration Over Code**: Behavior controlled through configuration objects
5. **Type Safety**: Pydantic models ensure data validation and type checking

## Future Enhancements

- [ ] Support for DOCX, HTML, and other formats
- [ ] Multi-threaded pipeline for parallel page processing
- [ ] Streaming API for large documents
- [ ] Table structure recognition
- [ ] Reading order detection
- [ ] Document chunking for RAG applications
- [ ] Export to multiple formats (Markdown, JSON, HTML)

## License

MIT License - See LICENSE file for details.

## Acknowledgments

Inspired by the [Docling](https://github.com/DS4SD/docling) project from IBM Research.

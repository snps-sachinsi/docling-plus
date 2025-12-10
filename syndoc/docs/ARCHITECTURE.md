# Architecture Documentation

## Overview

SynDoc is designed with a modular, layered architecture that separates concerns and enables easy extension. The design follows SOLID principles and common design patterns.

## Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DocumentConverter                       │
│                   (Facade/Orchestrator)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ coordinates
                     │
        ┌────────────┴─────────────┐
        │                          │
        ▼                          ▼
┌───────────────┐          ┌──────────────┐
│   Backend     │          │   Pipeline   │
│   (I/O)       │ provides │ (Processing) │
│               │ ────────>│              │
└───────┬───────┘  pages   └──────┬───────┘
        │                          │
        │                          │ uses
        │                          ▼
        │                  ┌──────────────┐
        │                  │    Models    │
        │                  │   (ML/AI)    │
        │                  └──────────────┘
        │
        ▼
┌───────────────┐
│  Data Models  │
│  (Pydantic)   │
└───────────────┘
```

## Layer Details

### 1. Data Models Layer

**Location**: `syndoc/datamodels/`

**Responsibility**: Define data structures and validation

**Key Components**:
- `InputDocument`: Metadata about source document
- `Page`: Single page with size and elements
- `DocumentElement`: Detected element (text, table, figure)
- `ConversionResult`: Complete conversion output
- `*Config`: Configuration classes (type-safe, validated)

**Design Patterns**:
- **Data Transfer Object (DTO)**: Pydantic models for serialization
- **Value Object**: Immutable value types like BoundingBox
- **Builder**: Nested configuration composition

**Benefits**:
- Type safety with runtime validation
- Automatic JSON serialization
- IDE autocomplete and type checking
- Clear contracts between layers

### 2. Backend Layer

**Location**: `syndoc/backends/`

**Responsibility**: Document I/O and format-specific operations

**Key Components**:
- `AbstractDocumentBackend`: Base interface
- `PaginatedDocumentBackend`: For page-based formats
- `DeclarativeDocumentBackend`: For structured formats
- `PdfBackend`: PDF implementation with PyPDFium2

**Design Patterns**:
- **Strategy**: Different backends for different formats
- **Template Method**: Common operations in abstract base
- **Adapter**: Wraps external libraries (pypdfium2)

**Extension Points**:
```python
class MyBackend(PaginatedDocumentBackend):
    def load(self): ...
    def load_page(self, page_no): ...
    def page_count(self): ...
    @classmethod
    def supported_formats(cls): ...
```

### 3. Pipeline Layer

**Location**: `syndoc/pipelines/`

**Responsibility**: Orchestrate document processing stages

**Key Components**:
- `BasePipeline`: Abstract pipeline with execute() template
- `PdfPipeline`: PDF-specific processing flow

**Processing Stages**:
1. Backend loading
2. Page extraction
3. Layout detection (optional)
4. OCR (optional)
5. Table structure (optional)
6. Reading order (optional)
7. Enrichment

**Design Patterns**:
- **Template Method**: execute() defines skeleton
- **Pipeline**: Sequential processing stages
- **Chain of Responsibility**: Stage-by-stage processing

**Extension Points**:
```python
class MyPipeline(BasePipeline):
    def _process_document(self, backend, result, raises_on_error):
        # Custom processing logic
        return result
```

### 4. Models Layer

**Location**: `syndoc/models/`

**Responsibility**: ML model integration and inference

**Key Components**:
- `BaseModel`: Abstract model interface
- `BaseLayoutModel`: Layout detection specialization
- `BaseOCRModel`: OCR specialization
- `ModelRegistry`: Factory for model instantiation

**Design Patterns**:
- **Strategy**: Different models for different tasks
- **Factory**: Registry-based model creation
- **Template Method**: load_model() + predict() pattern
- **Plugin Architecture**: Dynamic model registration

**Extension Points**:
```python
class MyModel(BaseLayoutModel):
    def load_model(self):
        # Load model artifacts
        pass
    
    def predict(self, images):
        # Run inference
        return predictions

# Register
ModelRegistry.register_layout_model("my_model", MyModel)
```

### 5. Converter Layer

**Location**: `syndoc/converter.py`

**Responsibility**: High-level API and orchestration

**Key Components**:
- `DocumentConverter`: Main entry point

**Design Patterns**:
- **Facade**: Simple interface hiding complexity
- **Factory Method**: Creates backends and pipelines
- **Builder**: Configuration-based construction

## Design Principles

### 1. Separation of Concerns

Each layer has a single responsibility:
- **Backends**: I/O only
- **Pipelines**: Processing flow
- **Models**: ML inference
- **Converter**: Orchestration

### 2. Dependency Inversion

High-level modules depend on abstractions:
```python
# Pipeline depends on abstract backend
class BasePipeline:
    def execute(self, backend: AbstractDocumentBackend):
        ...

# Not on concrete implementation
```

### 3. Open/Closed Principle

Open for extension, closed for modification:
- Add new backends without changing pipelines
- Add new models without changing pipelines
- Add new pipelines without changing converter

### 4. Interface Segregation

Small, focused interfaces:
- `AbstractDocumentBackend`: Basic operations
- `PaginatedDocumentBackend`: Adds pagination
- `DeclarativeDocumentBackend`: Adds direct conversion

### 5. Single Responsibility

Each class has one reason to change:
- `PdfBackend`: Changes only for PDF I/O
- `PdfPipeline`: Changes only for PDF processing
- `ModelConfig`: Changes only for model configuration

## Extension Patterns

### Adding a New Document Format

1. **Create Backend**:
```python
class DocxBackend(PaginatedDocumentBackend):
    def load(self): ...
    def load_page(self, page_no): ...
    # ...
```

2. **Create Pipeline** (if needed):
```python
class DocxPipeline(BasePipeline):
    def _process_document(self, backend, result, raises):
        # DOCX-specific processing
        return result
```

3. **Register in Converter**:
```python
def _create_backend(self, input_doc, source):
    if input_doc.format == InputFormat.DOCX:
        return DocxBackend(...)
    # ...
```

### Adding a New Model

1. **Implement Model**:
```python
class YoloLayoutModel(BaseLayoutModel):
    def load_model(self): ...
    def predict(self, images): ...
```

2. **Register**:
```python
ModelRegistry.register_layout_model("yolo", YoloLayoutModel)
```

3. **Use in Config**:
```python
config = ConversionConfig(
    pipeline=PipelineConfig(
        layout_model=ModelConfig(model_type="yolo", ...)
    )
)
```

### Adding a New Processing Stage

1. **Extend Pipeline**:
```python
class EnhancedPdfPipeline(PdfPipeline):
    def _process_page(self, backend, page):
        page = super()._process_page(backend, page)
        
        # Add custom stage
        with self._track_stage("custom_stage"):
            page = self._custom_processing(page)
        
        return page
```

## Configuration System

### Hierarchical Configuration

```
ConversionConfig
├── PipelineConfig
│   ├── layout_model: ModelConfig
│   ├── ocr_model: ModelConfig
│   └── table_model: ModelConfig
├── max_file_size
├── page_range
└── enable_profiling
```

### Configuration Sources

1. **Defaults**: Sensible defaults in Pydantic models
2. **Dictionary**: `from_config_dict()`
3. **Programmatic**: Direct instantiation
4. **Future**: YAML/JSON files

## Error Handling

### Error Capture vs Raise

```python
config = ConversionConfig(raises_on_error=False)  # Capture errors

result = converter.convert("doc.pdf")
if result.status == ConversionStatus.FAILURE:
    for error in result.errors:
        print(f"{error.stage}: {error.message}")
```

### Error Types

- **Validation Errors**: Pydantic catches at config time
- **I/O Errors**: Caught in backends, recorded in ErrorItem
- **Processing Errors**: Caught per-page, partial results returned
- **Model Errors**: Caught in pipeline stages

## Performance Optimization

### Profiling System

```python
config = ConversionConfig(enable_profiling=True)
result = converter.convert("doc.pdf")

print(result.profiling.total_time)
print(result.profiling.stage_times)
print(result.profiling.pages_per_second)
```

### Future Optimizations

1. **Multi-threading**: Parallel page processing
2. **Batching**: Batch model inference
3. **Caching**: Cache model predictions
4. **Streaming**: Process pages on-demand

## Testing Strategy

### Unit Tests

Test each layer independently:
- Data models: Validation, serialization
- Backends: I/O operations
- Pipelines: Processing logic
- Models: Inference (with mocks)

### Integration Tests

Test layer interactions:
- Backend + Pipeline
- Pipeline + Models
- End-to-end conversion

### Mock Objects

Provided mock models for testing:
- `MockLayoutModel`
- `MockOCRModel`
- `MockTableModel`

## Future Enhancements

### Short Term

- [ ] Multi-format support (DOCX, HTML)
- [ ] Real model integrations (DETR, Tesseract)
- [ ] Export formats (Markdown, HTML)
- [ ] CLI interface

### Medium Term

- [ ] Multi-threaded pipeline
- [ ] Streaming API
- [ ] Model download/caching
- [ ] Plugin system

### Long Term

- [ ] Distributed processing
- [ ] Cloud backend support
- [ ] Advanced table parsing
- [ ] Document understanding (semantic analysis)

## Comparison with Docling

| Aspect | Docling | SynDoc |
|--------|---------|---------|
| Architecture | Similar layered | Similar layered |
| Backends | Multiple formats | PDF (MVP) |
| Pipelines | Complex, threaded | Simple, extensible |
| Models | Integrated | Plugin-based |
| Configuration | Class-based | Pydantic-based |
| Focus | Production-ready | Educational, extensible |

## Summary

SynDoc provides a **clean, extensible architecture** for document parsing:

1. **Clear separation** of concerns across layers
2. **Abstract interfaces** for extension points
3. **Type-safe configuration** with Pydantic
4. **Factory patterns** for dynamic instantiation
5. **Template methods** for consistent behavior
6. **Plugin architecture** for models

The design enables easy extension while maintaining code quality and testability.

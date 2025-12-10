"""Example: Basic PDF conversion with SynDoc."""

import logging
from pathlib import Path

from syndoc import DocumentConverter, ConversionConfig, PipelineConfig, ModelConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def example_basic_conversion():
    """Example 1: Basic PDF conversion with default settings."""
    print("\n=== Example 1: Basic PDF Conversion ===\n")
    
    # Create converter with defaults
    converter = DocumentConverter()
    
    # Convert a PDF (you'll need to provide a real PDF path)
    pdf_path = "sample.pdf"
    
    if not Path(pdf_path).exists():
        print(f"Please provide a valid PDF at: {pdf_path}")
        return
    
    result = converter.convert(pdf_path)
    
    # Print results
    print(f"Status: {result.status.value}")
    print(f"Total pages: {len(result.pages)}")
    print(f"Errors: {len(result.errors)}")
    
    for page in result.pages[:3]:  # Show first 3 pages
        print(f"\nPage {page.page_no}:")
        print(f"  Size: {page.size.width} x {page.size.height}")
        print(f"  Elements: {len(page.elements)}")


def example_with_layout_detection():
    """Example 2: PDF conversion with layout detection (mock model)."""
    print("\n=== Example 2: With Layout Detection ===\n")
    
    # Configure with layout detection
    config = ConversionConfig(
        pipeline=PipelineConfig(
            do_layout_detection=True,
            layout_model=ModelConfig(
                model_type="mock",  # Using mock model for demo
                device="cpu",
                threshold=0.5,
                custom_params={"dpi": 144},
            ),
        ),
        enable_profiling=True,
    )
    
    converter = DocumentConverter(config=config)
    
    pdf_path = "sample.pdf"
    if not Path(pdf_path).exists():
        print(f"Please provide a valid PDF at: {pdf_path}")
        return
    
    result = converter.convert(pdf_path)
    
    print(f"Status: {result.status.value}")
    print(f"Total pages: {len(result.pages)}")
    
    # Show profiling info
    if result.profiling:
        print(f"\nPerformance:")
        print(f"  Total time: {result.profiling.total_time:.2f}s")
        print(f"  Pages/sec: {result.profiling.pages_per_second:.2f}")
        print(f"  Stage times:")
        for stage, time in result.profiling.stage_times.items():
            print(f"    {stage}: {time:.2f}s")
    
    # Show elements from first page
    if result.pages:
        page = result.pages[0]
        print(f"\nFirst page elements:")
        for elem in page.elements[:5]:
            print(f"  {elem.element_type.value}: {elem.bbox}")


def example_with_custom_config():
    """Example 3: Custom configuration from dictionary."""
    print("\n=== Example 3: Custom Configuration ===\n")
    
    config_dict = {
        "pipeline": {
            "do_layout_detection": True,
            "do_ocr": False,
            "layout_model": {
                "model_type": "mock",
                "device": "cpu",
                "threshold": 0.3,
                "batch_size": 1,
            },
        },
        "max_file_size": 50_000_000,  # 50 MB
        "page_range": (1, 10),  # First 10 pages only
        "enable_profiling": True,
    }
    
    converter = DocumentConverter.from_config_dict(config_dict)
    
    pdf_path = "sample.pdf"
    if not Path(pdf_path).exists():
        print(f"Please provide a valid PDF at: {pdf_path}")
        return
    
    result = converter.convert(pdf_path)
    
    print(f"Status: {result.status.value}")
    print(f"Pages processed: {len(result.pages)}")
    
    # Export to JSON
    json_str = result.to_json()
    print(f"\nJSON export length: {len(json_str)} characters")


def example_batch_conversion():
    """Example 4: Batch conversion of multiple PDFs."""
    print("\n=== Example 4: Batch Conversion ===\n")
    
    converter = DocumentConverter()
    
    pdf_files = [
        "document1.pdf",
        "document2.pdf",
        "document3.pdf",
    ]
    
    results = []
    for pdf_path in pdf_files:
        if not Path(pdf_path).exists():
            print(f"Skipping {pdf_path} (not found)")
            continue
        
        print(f"Converting {pdf_path}...")
        result = converter.convert(pdf_path)
        results.append(result)
        
        print(f"  Status: {result.status.value}")
        print(f"  Pages: {len(result.pages)}")
    
    print(f"\nTotal documents processed: {len(results)}")
    total_pages = sum(len(r.pages) for r in results)
    print(f"Total pages: {total_pages}")


def example_error_handling():
    """Example 5: Error handling."""
    print("\n=== Example 5: Error Handling ===\n")
    
    # Configure to NOT raise on errors
    config = ConversionConfig(
        raises_on_error=False,
    )
    
    converter = DocumentConverter(config=config)
    
    # Try to convert invalid file
    result = converter.convert("nonexistent.pdf")
    
    print(f"Status: {result.status.value}")
    print(f"Valid: {result.input.valid}")
    print(f"Errors: {len(result.errors)}")
    
    for error in result.errors:
        print(f"  {error.stage}: {error.message}")


if __name__ == "__main__":
    print("SynDoc Examples")
    print("=" * 50)
    
    # Run examples (uncomment the ones you want to try)
    example_basic_conversion()
    # example_with_layout_detection()
    # example_with_custom_config()
    # example_batch_conversion()
    # example_error_handling()

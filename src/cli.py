#!/usr/bin/env python
"""
ZeroTouch KYC OCR Extractor - Command Line Interface

Usage:
    python -m src.cli <file_path> [options]

Examples:
    python -m src.cli document.png
    python -m src.cli document.pdf --output json
    python -m src.cli document.png --output text --gpu
"""
import argparse
import json
import sys
import os

import torch

from src.core.extractor import DocumentExtractor

# Version should match the API version
__version__ = "1.0.0"


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="zerotouch-kyc",
        description="ZeroTouch KYC OCR Extractor - Extract fields from PAN, GST, and Bank documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python -m src.cli pan_card.png --output json\n  python -m src.cli gst_cert.pdf --gpu"
    )
    
    parser.add_argument(
        "file_path",
        help="Path to the image (PNG, JPG) or PDF file to process"
    )
    parser.add_argument(
        "-o", "--output",
        choices=["text", "json"],
        default="text",
        help="Output format: 'text' for human-readable, 'json' for machine-readable (default: text)"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration if CUDA is available"
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    return parser.parse_args()


def print_text_output(result: dict) -> None:
    """Format and print human-readable extraction results."""
    status = result.get("status", "unknown")
    doc_type = result.get("document_type", "UNKNOWN")
    
    print(f"\nDocument Type: {doc_type}")
    print(f"Status: {status}")
    
    if status == "success":
        fields = result.get("fields", {})
        if fields:
            print("\nExtracted Fields:")
            for key, field in fields.items():
                if isinstance(field, dict) and "value" in field:
                    value = field["value"]
                    confidence = field.get("confidence", 0.0)
                    print(f"  {key}: {value} (confidence: {confidence:.2f})")
                else:
                    print(f"  {key}: {field}")
        else:
            print("\nNo fields extracted.")
    else:
        message = result.get("message", "Unknown error")
        print(f"\nError: {message}", file=sys.stderr)


def main() -> int:
    """Main entry point. Returns exit code."""
    args = parse_args()
    
    # Validate file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File not found: {args.file_path}", file=sys.stderr)
        return 1
    
    # Determine GPU usage
    use_gpu = args.gpu and torch.cuda.is_available()
    if args.gpu and not torch.cuda.is_available():
        print("Warning: --gpu specified but CUDA is not available. Using CPU.", file=sys.stderr)
    
    # Initialize extractor and process
    try:
        extractor = DocumentExtractor(use_gpu=use_gpu)
        result = extractor.extract_from_file(args.file_path)
    except Exception as e:
        print(f"Error: Extraction failed: {e}", file=sys.stderr)
        return 2
    
    # Output results
    if args.output == "json":
        # JSON output: stable format for automation
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # Text output: human-readable format
        print_text_output(result)
    
    # Return appropriate exit code
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
